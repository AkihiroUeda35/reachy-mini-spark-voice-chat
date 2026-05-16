from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from itertools import islice
from typing import Any

import numpy as np
from numpy.typing import NDArray
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from scipy.spatial.transform import Rotation as R


logger = logging.getLogger(__name__)

WOBBLE_ORIGIN_POSE_ATTR = "_speech_wobble_origin_pose"

SR = 16_000
FRAME_MS = 20
HOP_MS = 50

SWAY_MASTER = 1.5
SENS_DB_OFFSET = 4.0
VAD_DB_ON = -35.0
VAD_DB_OFF = -45.0
VAD_ATTACK_MS = 40
VAD_RELEASE_MS = 250
ENV_FOLLOW_GAIN = 0.65

SWAY_F_PITCH = 2.2
SWAY_A_PITCH_DEG = 4.5
SWAY_F_YAW = 0.6
SWAY_A_YAW_DEG = 7.5
SWAY_F_ROLL = 1.3
SWAY_A_ROLL_DEG = 2.25
SWAY_F_X = 0.35
SWAY_A_X_MM = 4.5
SWAY_F_Y = 0.45
SWAY_A_Y_MM = 3.75
SWAY_F_Z = 0.25
SWAY_A_Z_MM = 2.25

SWAY_DB_LOW = -46.0
SWAY_DB_HIGH = -18.0
LOUDNESS_GAMMA = 0.9
SWAY_ATTACK_MS = 50
SWAY_RELEASE_MS = 250

FRAME = int(SR * FRAME_MS / 1000)
HOP = int(SR * HOP_MS / 1000)
ATTACK_FR = max(1, int(VAD_ATTACK_MS / HOP_MS))
RELEASE_FR = max(1, int(VAD_RELEASE_MS / HOP_MS))
SWAY_ATTACK_FR = max(1, int(SWAY_ATTACK_MS / HOP_MS))
SWAY_RELEASE_FR = max(1, int(SWAY_RELEASE_MS / HOP_MS))


def get_wobble_origin_pose(target: Any) -> NDArray[np.float64] | None:
    pose = getattr(target, WOBBLE_ORIGIN_POSE_ATTR, None)
    if pose is None:
        return None
    return np.array(pose, dtype=np.float64, copy=True)


def set_wobble_origin_pose(target: Any, pose: NDArray[np.float64] | None) -> None:
    if pose is None:
        if hasattr(target, WOBBLE_ORIGIN_POSE_ATTR):
            delattr(target, WOBBLE_ORIGIN_POSE_ATTR)
        return
    setattr(target, WOBBLE_ORIGIN_POSE_ATTR, np.array(pose, dtype=np.float64, copy=True))


def _rms_dbfs(audio: NDArray[np.float32]) -> float:
    samples = audio.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(samples * samples, dtype=np.float32) + 1e-12, dtype=np.float32)
    return float(20.0 * math.log10(float(rms) + 1e-12))


def _loudness_gain(db: float, offset: float = SENS_DB_OFFSET) -> float:
    value = (db + offset - SWAY_DB_LOW) / (SWAY_DB_HIGH - SWAY_DB_LOW)
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
    return value**LOUDNESS_GAMMA if LOUDNESS_GAMMA != 1.0 else value


def _to_float32_mono(audio: NDArray[Any]) -> NDArray[np.float32]:
    samples = np.asarray(audio)
    if samples.ndim == 0:
        return np.zeros(0, dtype=np.float32)

    if samples.ndim == 2:
        if samples.shape[0] <= 8 and samples.shape[0] <= samples.shape[1]:
            samples = np.mean(samples, axis=0)
        else:
            samples = np.mean(samples, axis=1)
    elif samples.ndim > 2:
        samples = np.mean(samples.reshape(samples.shape[0], -1), axis=0)

    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32, copy=False)

    info = np.iinfo(samples.dtype)
    scale = float(max(-info.min, info.max))
    return samples.astype(np.float32) / (scale if scale != 0.0 else 1.0)


def _resample_linear(audio: NDArray[np.float32], source_rate: int, target_rate: int) -> NDArray[np.float32]:
    if source_rate == target_rate or audio.size == 0:
        return audio

    target_length = int(round(audio.size * target_rate / source_rate))
    if target_length <= 1:
        return np.zeros(0, dtype=np.float32)

    source_positions = np.linspace(0.0, 1.0, num=audio.size, dtype=np.float32, endpoint=True)
    target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32, endpoint=True)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)


class SwayRollRT:
    def __init__(self, rng_seed: int = 7):
        self._logger = logging.getLogger(f"{__name__}.sway")
        self.samples: deque[float] = deque(maxlen=10 * SR)
        self.carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0
        self.sway_env = 0.0
        self.sway_up = 0
        self.sway_down = 0

        rng = np.random.default_rng(int(rng_seed))
        self.phase_pitch = float(rng.random() * 2 * math.pi)
        self.phase_yaw = float(rng.random() * 2 * math.pi)
        self.phase_roll = float(rng.random() * 2 * math.pi)
        self.phase_x = float(rng.random() * 2 * math.pi)
        self.phase_y = float(rng.random() * 2 * math.pi)
        self.phase_z = float(rng.random() * 2 * math.pi)
        self.t = 0.0

    def reset(self) -> None:
        self.samples.clear()
        self.carry = np.zeros(0, dtype=np.float32)
        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0
        self.sway_env = 0.0
        self.sway_up = 0
        self.sway_down = 0
        self.t = 0.0

    def feed(self, pcm: NDArray[Any], sample_rate: int | None) -> list[dict[str, float]]:
        source_rate = SR if sample_rate is None else int(sample_rate)
        samples = _to_float32_mono(pcm)
        if samples.size == 0:
            return []
        if source_rate != SR:
            samples = _resample_linear(samples, source_rate, SR)
            if samples.size == 0:
                return []

        if self.carry.size:
            self.carry = np.concatenate([self.carry, samples])
        else:
            self.carry = samples

        out: list[dict[str, float]] = []
        while self.carry.size >= HOP:
            hop = self.carry[:HOP]
            self.carry = self.carry[HOP:]
            self.samples.extend(hop.tolist())
            if len(self.samples) < FRAME:
                self.t += HOP_MS / 1000.0
                continue

            frame = np.fromiter(
                islice(self.samples, len(self.samples) - FRAME, len(self.samples)),
                dtype=np.float32,
                count=FRAME,
            )
            db = _rms_dbfs(frame)
            vad_before = self.vad_on

            if db >= VAD_DB_ON:
                self.vad_above += 1
                self.vad_below = 0
                if not self.vad_on and self.vad_above >= ATTACK_FR:
                    self.vad_on = True
            elif db <= VAD_DB_OFF:
                self.vad_below += 1
                self.vad_above = 0
                if self.vad_on and self.vad_below >= RELEASE_FR:
                    self.vad_on = False

            if vad_before != self.vad_on:
                self._logger.debug(
                    "Speech sway VAD %s db=%.1f env=%.3f t=%.3fs carry=%d",
                    "on" if self.vad_on else "off",
                    db,
                    self.sway_env,
                    self.t,
                    self.carry.size,
                )

            if self.vad_on:
                self.sway_up = min(SWAY_ATTACK_FR, self.sway_up + 1)
                self.sway_down = 0
            else:
                self.sway_down = min(SWAY_RELEASE_FR, self.sway_down + 1)
                self.sway_up = 0

            up = self.sway_up / SWAY_ATTACK_FR
            down = 1.0 - (self.sway_down / SWAY_RELEASE_FR)
            target = up if self.vad_on else down
            self.sway_env += ENV_FOLLOW_GAIN * (target - self.sway_env)
            if self.sway_env < 0.0:
                self.sway_env = 0.0
            elif self.sway_env > 1.0:
                self.sway_env = 1.0

            loud = _loudness_gain(db) * SWAY_MASTER
            env = self.sway_env
            self.t += HOP_MS / 1000.0

            pitch = math.radians(SWAY_A_PITCH_DEG) * loud * env * math.sin(2 * math.pi * SWAY_F_PITCH * self.t + self.phase_pitch)
            yaw = math.radians(SWAY_A_YAW_DEG) * loud * env * math.sin(2 * math.pi * SWAY_F_YAW * self.t + self.phase_yaw)
            roll = math.radians(SWAY_A_ROLL_DEG) * loud * env * math.sin(2 * math.pi * SWAY_F_ROLL * self.t + self.phase_roll)
            x_mm = SWAY_A_X_MM * loud * env * math.sin(2 * math.pi * SWAY_F_X * self.t + self.phase_x)
            y_mm = SWAY_A_Y_MM * loud * env * math.sin(2 * math.pi * SWAY_F_Y * self.t + self.phase_y)
            z_mm = SWAY_A_Z_MM * loud * env * math.sin(2 * math.pi * SWAY_F_Z * self.t + self.phase_z)

            out.append(
                {
                    "pitch_rad": pitch,
                    "yaw_rad": yaw,
                    "roll_rad": roll,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "z_mm": z_mm,
                }
            )

        return out


class HeadWobbler:
    def __init__(
        self,
        set_target_head_pose: Callable[[NDArray[np.float64]], None],
        get_current_head_pose: Callable[[], NDArray[np.float64]] | None = None,
        get_origin_head_pose: Callable[[], NDArray[np.float64] | None] | None = None,
        *,
        movement_latency_s: float = 0.05,
        reset_center_ratio: float = 1.0,
    ) -> None:
        self._logger = logging.getLogger(f"{__name__}.wobbler")
        self._set_target_head_pose = set_target_head_pose
        self._get_current_head_pose = get_current_head_pose
        self._get_origin_head_pose = get_origin_head_pose
        self._movement_latency_s = max(0.0, movement_latency_s)
        self._reset_center_ratio = min(1.0, max(0.0, float(reset_center_ratio)))
        self._base_ts: float | None = None
        self._base_pose: NDArray[np.float64] | None = None
        self._hops_done = 0
        self._generation = 0
        self._reset_after_audio = False
        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._idle_event = threading.Event()
        self._idle_event.set()
        self._thread: threading.Thread | None = None
        self._audio_queue: queue.Queue[tuple[int, int, NDArray[Any], float]] = queue.Queue()
        self._sway = SwayRollRT()

    def feed_pcm(self, pcm: NDArray[Any], sample_rate: int, *, start_delay_s: float = 0.0) -> None:
        self._ensure_started()
        chunk = np.array(pcm, copy=True)
        with self._state_lock:
            generation = self._generation
            self._reset_after_audio = False
        self._idle_event.clear()
        self._audio_queue.put((generation, sample_rate, chunk, max(0.0, start_delay_s)))
        duration_s = 0.0
        if sample_rate > 0:
            duration_s = len(np.asarray(chunk).reshape(-1)) / max(1, sample_rate)
        self._logger.debug(
            "Head wobble queued chunk gen=%d sr=%d duration=%.3fs queue=%d start_delay=%.3fs",
            generation,
            sample_rate,
            duration_s,
            self._audio_queue.qsize(),
            max(0.0, start_delay_s),
        )

    def request_reset_after_current_audio(self) -> None:
        should_reset_now = False
        with self._state_lock:
            self._reset_after_audio = True
            should_reset_now = self._base_ts is None and self._audio_queue.empty()
            base_ts = self._base_ts
            hops_done = self._hops_done
        self._logger.debug(
            "Head wobble reset requested queue_empty=%s base_ts_set=%s hops_done=%d",
            self._audio_queue.empty(),
            base_ts is not None,
            hops_done,
        )
        if should_reset_now:
            self.reset()

    def finish(self, timeout_s: float | None = None) -> bool:
        if self._thread is None:
            self.reset()
            return True
        self.request_reset_after_current_audio()
        if timeout_s is None:
            self._logger.debug("Head wobble waiting for idle without timeout")
            self._idle_event.wait()
            self._logger.debug("Head wobble reached idle state")
            return True
        finished = self._idle_event.wait(timeout=max(0.0, timeout_s))
        if not finished:
            with self._state_lock:
                base_ts = self._base_ts
                hops_done = self._hops_done
                reset_after_audio = self._reset_after_audio
            self._logger.debug(
                "Head wobble finish timed out timeout=%.3fs queue=%d base_ts=%s hops_done=%d reset_after_audio=%s",
                timeout_s,
                self._audio_queue.qsize(),
                "set" if base_ts is not None else "unset",
                hops_done,
                reset_after_audio,
            )
        return finished

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if not self._idle_event.is_set():
            self.reset()

    def reset(self) -> None:
        target_pose = self._pose_toward_origin(self._base_pose, self._capture_origin_pose(), self._reset_center_ratio)
        with self._state_lock:
            self._generation += 1
            self._base_ts = None
            self._base_pose = None
            self._hops_done = 0
            self._reset_after_audio = False

        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._audio_queue.task_done()

        with self._sway_lock:
            self._sway.reset()

        self._set_target_head_pose(target_pose)
        self._idle_event.set()
        self._logger.debug("Head wobble reset complete")

    def _pose_toward_origin(
        self,
        pose: NDArray[np.float64] | None,
        origin_pose: NDArray[np.float64] | None,
        center_ratio: float,
    ) -> NDArray[np.float64]:
        anchor_pose = create_head_pose(degrees=False) if origin_pose is None else origin_pose
        if pose is None:
            return np.array(anchor_pose, dtype=np.float64, copy=True)
        pose_ratio = 1.0 - min(1.0, max(0.0, float(center_ratio)))
        base_translation = np.array(pose[:3, 3], dtype=np.float64, copy=True)
        origin_translation = np.array(anchor_pose[:3, 3], dtype=np.float64, copy=True)
        translation = origin_translation + (base_translation - origin_translation) * pose_ratio
        base_rotation = R.from_matrix(np.asarray(pose[:3, :3], dtype=np.float64)).as_euler("xyz")
        origin_rotation = R.from_matrix(np.asarray(anchor_pose[:3, :3], dtype=np.float64)).as_euler("xyz")
        rotation = origin_rotation + (base_rotation - origin_rotation) * pose_ratio
        return create_head_pose(*translation, *rotation, degrees=False)

    def _ensure_started(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._working_loop, name="reachy-head-wobbler", daemon=True)
        self._thread.start()
        self._logger.debug("Head wobble thread started")

    def _working_loop(self) -> None:
        hop_dt = HOP_MS / 1000.0
        while not self._stop_event.is_set():
            try:
                chunk_generation, sample_rate, chunk, start_delay_s = self._audio_queue.get(timeout=hop_dt)
            except queue.Empty:
                if self._should_reset_after_audio(hop_dt):
                    self.reset()
                continue

            try:
                with self._state_lock:
                    current_generation = self._generation
                if chunk_generation != current_generation:
                    continue

                if self._base_ts is None:
                    with self._state_lock:
                        if self._base_ts is None:
                            self._base_ts = time.monotonic() + start_delay_s
                if self._base_pose is None:
                    self._base_pose = self._capture_base_pose()

                with self._sway_lock:
                    results = self._sway.feed(chunk, sample_rate)

                self._logger.debug(
                    "Head wobble processed chunk gen=%d sr=%d sway_frames=%d queue=%d",
                    chunk_generation,
                    sample_rate,
                    len(results),
                    self._audio_queue.qsize(),
                )

                index = 0
                while index < len(results) and not self._stop_event.is_set():
                    with self._state_lock:
                        if self._generation != current_generation:
                            break
                        base_ts = self._base_ts
                        hops_done = self._hops_done

                    if base_ts is None:
                        base_ts = time.monotonic()
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = base_ts
                                hops_done = self._hops_done

                    target = base_ts + self._movement_latency_s + hops_done * hop_dt
                    now = time.monotonic()
                    if now - target >= hop_dt:
                        lag_hops = int((now - target) / hop_dt)
                        drop = min(lag_hops, len(results) - index - 1)
                        if drop > 0:
                            with self._state_lock:
                                self._hops_done += drop
                            index += drop
                            continue

                    if target > now:
                        time.sleep(target - now)

                    result = results[index]
                    offset_pose = create_head_pose(
                        result["x_mm"] / 1000.0,
                        result["y_mm"] / 1000.0,
                        result["z_mm"] / 1000.0,
                        result["roll_rad"],
                        result["pitch_rad"],
                        result["yaw_rad"],
                        degrees=False,
                    )
                    base_pose = self._base_pose if self._base_pose is not None else create_head_pose(degrees=False)
                    pose = compose_world_offset(base_pose, offset_pose, reorthonormalize=False)
                    with self._state_lock:
                        if self._generation != current_generation:
                            break
                    self._set_target_head_pose(pose)
                    with self._state_lock:
                        self._hops_done += 1
                    index += 1
            finally:
                self._audio_queue.task_done()
        self._logger.debug("Head wobble thread exited")

    def _should_reset_after_audio(self, hop_dt: float) -> bool:
        with self._state_lock:
            if not self._reset_after_audio or self._base_ts is None:
                return False
            if not self._audio_queue.empty():
                return False
            reset_at = self._base_ts + self._movement_latency_s + self._hops_done * hop_dt
        should_reset = time.monotonic() >= reset_at
        if should_reset:
            self._logger.debug(
                "Head wobble reached scheduled reset point hops_done=%d",
                self._hops_done,
            )
        return should_reset

    def _capture_base_pose(self) -> NDArray[np.float64]:
        if self._get_current_head_pose is None:
            return create_head_pose(degrees=False)
        try:
            return np.array(self._get_current_head_pose(), copy=True)
        except Exception:
            logger.debug("Falling back to neutral head pose for speech wobble base", exc_info=True)
            return create_head_pose(degrees=False)

    def _capture_origin_pose(self) -> NDArray[np.float64] | None:
        if self._get_origin_head_pose is None:
            return None
        try:
            pose = self._get_origin_head_pose()
        except Exception:
            logger.debug("Falling back to neutral head pose for speech wobble origin", exc_info=True)
            return None
        if pose is None:
            return None
        return np.array(pose, copy=True)