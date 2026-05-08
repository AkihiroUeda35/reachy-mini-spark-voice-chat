from __future__ import annotations

import asyncio
import importlib
import json
import logging
import random
import subprocess
import sys
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from langchain_core.tools import BaseTool, StructuredTool, tool
from PIL import Image
from pydantic import BaseModel, Field
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from reachy_mini_dances_library.dance_move import DanceMove

from lib.jma_weather_tool import get_jma_weather_tool


logger = logging.getLogger(__name__)


DEFAULT_EMOTION_DATASET = "pollen-robotics/reachy-mini-emotions-library"
DANCE_PACKAGE_SPEC = "reachy-mini-dances-library>=0.2.1"
DANCE_IMPORT_NAME = "reachy_mini_dances_library.collection.dance"
VISION_PACKAGE_SPEC = "opencv-python-headless>=4.10.0"
VISION_IMPORT_NAME = "cv2"
GUI_TOOL_NAMES = [
    "dance",
    "stop_dance",
    "play_emotion",
    "stop_emotion",
    "camera",
    "idle_do_nothing",
    "head_tracking",
    "move_head",
    "get_jma_weather_tool",
    "get_current_time_tool",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


@dataclass
class BackgroundToolRecord:
    tool_id: str
    tool_name: str
    started_at: str
    status: str = "running"
    completed_at: str | None = None
    result: Any = None
    error: str | None = None
    task: asyncio.Task[Any] | None = field(default=None, repr=False)


class MoveHeadArgs(BaseModel):
    direction: Literal["left", "right", "up", "down", "front"]


class DanceArgs(BaseModel):
    move: str | None = Field(default=None, description="Name of the dance move. Omit for a random move.")
    repeat: int = Field(default=1, ge=1, le=10, description="How many times to repeat the move.")


class StopMoveArgs(BaseModel):
    dummy: bool = Field(description="Dummy boolean. Set to true.")


class PlayEmotionArgs(BaseModel):
    emotion: str | None = Field(default=None, description="Name of the emotion to play. Omit for a random emotion.")


class CameraArgs(BaseModel):
    question: str = Field(description="The question to ask about the latest camera frame.")


class HeadTrackingArgs(BaseModel):
    start: bool


class IdleArgs(BaseModel):
    reason: str | None = Field(default=None, description="Optional reason for staying idle.")


class TaskStatusArgs(BaseModel):
    tool_id: str | None = Field(default=None, description="Specific background tool ID to inspect.")


class TaskCancelArgs(BaseModel):
    tool_id: str = Field(description="Background tool ID to cancel.")


@tool
def get_current_time_tool() -> str:
    """Get the current local date and time on this machine."""
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


class ReachyToolRuntime:
    def __init__(
        self,
        robot: ReachyMini,
        *,
        data_dir: Path,
        motion_duration_s: float = 0.8,
        chat_base_url: str,
        chat_api_key: str,
        chat_model: str,
        chat_timeout_s: float = 30.0,
        auto_install_optional_deps: bool = True,
    ):
        self.robot = robot
        self.data_dir = data_dir
        self.motion_duration_s = motion_duration_s
        self.head_tracking_enabled = False
        self.chat_base_url = chat_base_url.rstrip("/")
        self.chat_api_key = chat_api_key
        self.chat_model = chat_model
        self.chat_timeout_s = chat_timeout_s
        self.auto_install_optional_deps = auto_install_optional_deps
        self._emotion_moves: RecordedMoves | None = None
        self._emotion_unavailable = False
        self._background_tasks: dict[str, BackgroundToolRecord] = {}
        self._motion_tool_names = {"dance", "play_emotion"}
        self._dance_moves: dict[str, Any] | None = None
        self._dance_unavailable = False
        self._cv2: Any | None = None
        self._vision_unavailable = False
        self._head_tracking_task: asyncio.Task[None] | None = None

    def list_available_dances(self) -> list[str]:
        return sorted(self._load_dance_moves().keys())

    def list_available_emotions(self) -> list[str]:
        recorded_moves = self._load_recorded_emotions()
        if recorded_moves is None:
            return []
        try:
            return sorted(recorded_moves.list_moves())
        except Exception:
            logger.exception("Failed to enumerate emotion moves")
            return []

    def _load_recorded_emotions(self) -> RecordedMoves | None:
        if self._emotion_unavailable:
            return None
        if self._emotion_moves is not None:
            return self._emotion_moves

        try:
            self._emotion_moves = RecordedMoves(DEFAULT_EMOTION_DATASET)
        except Exception:
            logger.exception("Failed to load Reachy emotion dataset")
            self._emotion_unavailable = True
            return None
        return self._emotion_moves

    def _install_optional_package(self, package_spec: str, import_name: str) -> bool:
        if not self.auto_install_optional_deps:
            return False

        logger.info("Installing optional dependency %s for %s", package_spec, import_name)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error("Optional dependency install failed: %s", result.stderr.strip() or result.stdout.strip())
            return False
        importlib.invalidate_caches()
        return True

    def _load_dance_moves(self) -> dict[str, Any]:
        if self._dance_unavailable:
            return {}
        if self._dance_moves is not None:
            return self._dance_moves

        try:
            dance_module = importlib.import_module(DANCE_IMPORT_NAME)
        except ImportError:
            if not self._install_optional_package(DANCE_PACKAGE_SPEC, DANCE_IMPORT_NAME):
                self._dance_unavailable = True
                return {}
            try:
                dance_module = importlib.import_module(DANCE_IMPORT_NAME)
            except ImportError:
                self._dance_unavailable = True
                return {}

        self._dance_moves = dict(getattr(dance_module, "AVAILABLE_MOVES", {}) or {})
        return self._dance_moves

    def _load_cv2(self) -> Any | None:
        if self._vision_unavailable:
            return None
        if self._cv2 is not None:
            return self._cv2

        try:
            self._cv2 = importlib.import_module(VISION_IMPORT_NAME)
        except ImportError:
            if not self._install_optional_package(VISION_PACKAGE_SPEC, VISION_IMPORT_NAME):
                self._vision_unavailable = True
                return None
            try:
                self._cv2 = importlib.import_module(VISION_IMPORT_NAME)
            except ImportError:
                self._vision_unavailable = True
                return None
        return self._cv2

    def _face_cascade(self) -> Any | None:
        cv2 = self._load_cv2()
        if cv2 is None:
            return None
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.is_file():
            return None
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return None
        return cascade

    def _analyze_frame(self, frame: np.ndarray) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "mean_brightness": round(float(frame.mean()), 2),
            "dominant_bgr": [int(channel) for channel in frame.reshape(-1, 3).mean(axis=0)],
            "face_count": 0,
            "faces": [],
            "primary_face_center": None,
        }

        cv2 = self._load_cv2()
        cascade = self._face_cascade()
        if cv2 is None or cascade is None:
            observation["vision_backend"] = "unavailable"
            return observation

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        observation["vision_backend"] = "opencv-haar"
        observation["face_count"] = int(len(faces))

        face_entries: list[dict[str, int]] = []
        largest_face: tuple[int, int, int, int] | None = None
        largest_area = -1
        for (x, y, w, h) in faces:
            entry = {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "cx": int(x + w / 2), "cy": int(y + h / 2)}
            face_entries.append(entry)
            area = int(w * h)
            if area > largest_area:
                largest_area = area
                largest_face = (int(x), int(y), int(w), int(h))

        observation["faces"] = face_entries[:5]
        if largest_face is not None:
            x, y, w, h = largest_face
            observation["primary_face_center"] = {"x": int(x + w / 2), "y": int(y + h / 2)}
            observation["primary_face_size"] = {"w": w, "h": h}
        return observation

    async def _answer_camera_question(self, question: str, observation: dict[str, Any]) -> str:
        prompt = (
            "You are answering a question about a camera frame using only structured computer-vision observations. "
            "Do not invent details that are not present. If the observation is insufficient, say that clearly in Japanese.\n\n"
            f"Question: {question}\n"
            f"Observation JSON: {json.dumps(observation, ensure_ascii=False)}"
        )
        model = ChatOpenAI(
            model=self.chat_model,
            base_url=self.chat_base_url,
            api_key=SecretStr(self.chat_api_key),
            temperature=0.1,
            max_completion_tokens=180,
            timeout=self.chat_timeout_s,
            use_responses_api=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        response = await model.ainvoke(prompt)
        answer = str(getattr(response, "content", "") or "").strip()
        if answer:
            return answer

        if observation.get("face_count", 0) > 0:
            return f"画像では顔を {observation['face_count']} 人分検出しました。質問に答えるには、この観測だけでは足りないかもしれません。"
        return "画像を解析しましたが、観測できた情報だけではその質問に十分答えられません。"

    async def _head_tracking_loop(self) -> None:
        logger.info("Head tracking loop started")
        while self.head_tracking_enabled:
            try:
                frame = await asyncio.to_thread(self.robot.media.get_frame)
                if frame is None:
                    await asyncio.sleep(0.2)
                    continue

                observation = await asyncio.to_thread(self._analyze_frame, frame)
                center = observation.get("primary_face_center")
                if isinstance(center, dict):
                    u = max(1, min(int(center.get("x", 1)), frame.shape[1] - 1))
                    v = max(1, min(int(center.get("y", 1)), frame.shape[0] - 1))
                    await asyncio.to_thread(self.robot.look_at_image, u, v, 0.25, True)
                await asyncio.sleep(0.35)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Head tracking iteration failed")
                await asyncio.sleep(0.5)
        logger.info("Head tracking loop stopped")

    async def move_head(self, direction: str) -> dict[str, Any]:
        deltas = {
            "left": (0, 0, 0, 0, 0, 40),
            "right": (0, 0, 0, 0, 0, -40),
            "up": (0, 0, 0, 0, -30, 0),
            "down": (0, 0, 0, 0, 30, 0),
            "front": (0, 0, 0, 0, 0, 0),
        }
        target = create_head_pose(*deltas.get(direction, deltas["front"]), degrees=True)
        logger.info("Tool call: move_head direction=%s", direction)
        await asyncio.to_thread(
            self.robot.goto_target,
            head=target,
            duration=self.motion_duration_s,
            body_yaw=None,
        )
        return {"status": f"looking {direction}"}

    async def dance(self, move: str | None, repeat: int) -> dict[str, Any]:
        available_moves = self._load_dance_moves()
        if not available_moves:
            return {"error": "Dance system not available. Install the Reachy Mini dance library to enable this tool."}

        move_name = move or random.choice(list(available_moves.keys()))
        if move_name not in available_moves:
            return {"error": f"Unknown dance move '{move_name}'. Available: {sorted(available_moves.keys())}"}

        async def runner() -> dict[str, Any]:
            for _ in range(repeat):
                dance_move = DanceMove(move_name)
                await self.robot.async_play_move(dance_move, initial_goto_duration=0.25)
            return {"status": "completed", "move": move_name, "repeat": repeat}

        logger.info("Tool call: dance move=%s repeat=%d", move_name, repeat)
        return self._start_background_tool("dance", runner)

    async def stop_dance(self) -> dict[str, Any]:
        logger.info("Tool call: stop_dance")
        await self._cancel_tools({"dance"})
        return {"status": "stopped dance and cleared queue"}

    async def play_emotion(self, emotion: str | None) -> dict[str, Any]:
        recorded_moves = self._load_recorded_emotions()
        if recorded_moves is None:
            return {"error": "Emotion system not available."}

        emotion_names = recorded_moves.list_moves()
        if not emotion_names:
            return {"error": "No emotions are currently available."}

        emotion_name = emotion or random.choice(emotion_names)
        if emotion_name not in emotion_names:
            return {"error": f"Unknown emotion '{emotion_name}'. Available: {emotion_names}"}

        async def runner() -> dict[str, Any]:
            move = recorded_moves.get(emotion_name)
            await self.robot.async_play_move(move, initial_goto_duration=0.2)
            return {"status": "completed", "emotion": emotion_name}

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)
        return self._start_background_tool("play_emotion", runner)

    async def stop_emotion(self) -> dict[str, Any]:
        logger.info("Tool call: stop_emotion")
        await self._cancel_tools({"play_emotion"})
        return {"status": "stopped emotion and cleared queue"}

    async def camera(self, question: str) -> dict[str, Any]:
        logger.info("Tool call: camera question=%s", question)
        frame = await asyncio.to_thread(self.robot.media.get_frame)
        if frame is None:
            return {"error": "Camera is not available on this Reachy Mini instance."}

        snapshot_dir = self.data_dir / "camera"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        rgb = frame[:, :, ::-1]
        await asyncio.to_thread(Image.fromarray(rgb).save, snapshot_path)
        observation = await asyncio.to_thread(self._analyze_frame, frame)
        answer = await self._answer_camera_question(question, observation)
        return {
            "status": "captured",
            "question": question,
            "image_path": str(snapshot_path),
            "observation": observation,
            "answer": answer,
        }

    async def head_tracking(self, start: bool) -> dict[str, Any]:
        self.head_tracking_enabled = start
        status = "started" if start else "stopped"
        logger.info("Tool call: head_tracking %s", status)
        if start:
            if self._load_cv2() is None:
                return {"error": "Vision backend is not available. Install OpenCV to enable head tracking."}
            if self._head_tracking_task is None or self._head_tracking_task.done():
                self._head_tracking_task = asyncio.create_task(self._head_tracking_loop(), name="reachy-head-tracking")
        else:
            if self._head_tracking_task is not None:
                self._head_tracking_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._head_tracking_task
                self._head_tracking_task = None
        return {
            "status": f"head tracking {status}",
            "message": "Vision tracking uses local OpenCV face detection and Reachy's look_at_image API.",
        }

    async def idle_do_nothing(self, reason: str | None) -> dict[str, Any]:
        final_reason = reason or "idle turn"
        logger.info("Tool call: idle_do_nothing reason=%s", final_reason)
        return {"status": "idle", "reason": final_reason}

    async def task_status(self, tool_id: str | None) -> dict[str, Any]:
        logger.info("Tool call: task_status tool_id=%s", tool_id)
        if tool_id:
            record = self._background_tasks.get(tool_id)
            if record is None:
                return {"error": f"Tool {tool_id} not found."}
            return self._record_payload(record)

        running = [record for record in self._background_tasks.values() if record.status == "running"]
        if not running:
            return {"status": "idle", "message": "No tools running in the background."}

        return {
            "status": "running",
            "count": len(running),
            "message": f"{len(running)} tool(s) running in the background.",
            "tools": [self._record_payload(record) for record in running],
        }

    async def task_cancel(self, tool_id: str) -> dict[str, Any]:
        logger.info("Tool call: task_cancel tool_id=%s", tool_id)
        record = self._background_tasks.get(tool_id)
        if record is None:
            return {"error": f"Tool {tool_id} not found."}
        if record.status != "running":
            return {
                "status": record.status,
                "message": f"Tool '{record.tool_name}' is not running (status: {record.status}).",
                "tool_id": tool_id,
            }

        await self._cancel_task(record)
        return {
            "status": "cancelled",
            "tool_id": tool_id,
            "message": f"Cancelled background tool '{record.tool_name}'.",
        }

    async def shutdown(self) -> None:
        if self._head_tracking_task is not None:
            self._head_tracking_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._head_tracking_task
            self._head_tracking_task = None
        for record in list(self._background_tasks.values()):
            if record.status == "running":
                await self._cancel_task(record)

    def _start_background_tool(self, tool_name: str, runner: Any) -> dict[str, Any]:
        tool_id = uuid.uuid4().hex[:8]
        record = BackgroundToolRecord(tool_id=tool_id, tool_name=tool_name, started_at=_now_iso())

        async def wrapped() -> None:
            try:
                record.result = await runner()
                record.status = "completed"
            except asyncio.CancelledError:
                record.status = "cancelled"
                raise
            except Exception as exc:
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
                logger.exception("Background tool failed: %s", tool_name)
            finally:
                record.completed_at = _now_iso()

        record.task = asyncio.create_task(wrapped(), name=f"reachy-tool:{tool_name}:{tool_id}")
        self._background_tasks[tool_id] = record
        return {
            "status": "queued",
            "tool_id": tool_id,
            "name": tool_name,
            "message": f"Started background tool '{tool_name}'.",
        }

    def _record_payload(self, record: BackgroundToolRecord) -> dict[str, Any]:
        payload = {
            "tool_id": record.tool_id,
            "name": record.tool_name,
            "status": record.status,
            "started_at": record.started_at,
        }
        if record.completed_at is not None:
            payload["completed_at"] = record.completed_at
        if record.result is not None:
            payload["result"] = record.result
        if record.error is not None:
            payload["error"] = record.error
        return payload

    async def _cancel_task(self, record: BackgroundToolRecord) -> None:
        if record.tool_name in self._motion_tool_names:
            self.robot.cancel_move()
        if record.task is not None:
            record.task.cancel()
            with suppress(asyncio.CancelledError):
                await record.task
        record.status = "cancelled"
        record.completed_at = _now_iso()

    async def _cancel_tools(self, names: set[str]) -> None:
        for record in list(self._background_tasks.values()):
            if record.status == "running" and record.tool_name in names:
                await self._cancel_task(record)


def _tool_result_description(prefix: str, values: list[str]) -> str:
    if not values:
        return prefix
    return f"{prefix}\nAvailable values: {', '.join(values)}"


def build_langchain_tools(runtime: ReachyToolRuntime, enabled_tool_names: list[str]) -> list[BaseTool]:
    dance_description = _tool_result_description(
        "Play a named or random dance move once or repeatedly in the background.",
        runtime.list_available_dances(),
    )
    emotion_description = _tool_result_description(
        "Play a named or random pre-recorded emotion in the background.",
        runtime.list_available_emotions(),
    )

    async def stop_dance_wrapper(dummy: bool) -> dict[str, Any]:
        _ = dummy
        return await runtime.stop_dance()

    async def stop_emotion_wrapper(dummy: bool) -> dict[str, Any]:
        _ = dummy
        return await runtime.stop_emotion()

    tool_factories: dict[str, Any] = {
        "move_head": lambda: StructuredTool.from_function(
            coroutine=runtime.move_head,
            name="move_head",
            description="Move your head in a given direction: left, right, up, down or front.",
            args_schema=MoveHeadArgs,
        ),
        "dance": lambda: StructuredTool.from_function(
            coroutine=runtime.dance,
            name="dance",
            description=dance_description,
            args_schema=DanceArgs,
        ),
        "stop_dance": lambda: StructuredTool.from_function(
            coroutine=stop_dance_wrapper,
            name="stop_dance",
            description="Stop the current dance move.",
            args_schema=StopMoveArgs,
        ),
        "play_emotion": lambda: StructuredTool.from_function(
            coroutine=runtime.play_emotion,
            name="play_emotion",
            description=emotion_description,
            args_schema=PlayEmotionArgs,
        ),
        "stop_emotion": lambda: StructuredTool.from_function(
            coroutine=stop_emotion_wrapper,
            name="stop_emotion",
            description="Stop the current emotion.",
            args_schema=StopMoveArgs,
        ),
        "camera": lambda: StructuredTool.from_function(
            coroutine=runtime.camera,
            name="camera",
            description="Take a picture with the camera and answer a question about it.",
            args_schema=CameraArgs,
        ),
        "head_tracking": lambda: StructuredTool.from_function(
            coroutine=runtime.head_tracking,
            name="head_tracking",
            description="Toggle head tracking state.",
            args_schema=HeadTrackingArgs,
        ),
        "idle_do_nothing": lambda: StructuredTool.from_function(
            coroutine=runtime.idle_do_nothing,
            name="idle_do_nothing",
            description="Stay still and silent during an idle turn.",
            args_schema=IdleArgs,
        ),
        "get_jma_weather_tool": lambda: get_jma_weather_tool,
        "get_current_time_tool": lambda: get_current_time_tool,
        "task_status": lambda: StructuredTool.from_function(
            coroutine=runtime.task_status,
            name="task_status",
            description="Check the status of background tool tasks.",
            args_schema=TaskStatusArgs,
        ),
        "task_cancel": lambda: StructuredTool.from_function(
            coroutine=runtime.task_cancel,
            name="task_cancel",
            description="Cancel a running background tool task.",
            args_schema=TaskCancelArgs,
        ),
    }

    names = list(dict.fromkeys([*enabled_tool_names, "task_status", "task_cancel"]))
    tools: list[BaseTool] = []
    for name in names:
        factory = tool_factories.get(name)
        if factory is None:
            logger.warning("Skipping unknown Reachy tool '%s'", name)
            continue
        tools.append(factory())
    return tools