from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import SupportsFloat
from urllib.parse import urlsplit

import gradio as gr
import httpx

import reachy_conversation_tools
from state import DEFAULT_CHARACTER_PROMPT, DEFAULT_QWEN_VOICE, DEFAULT_TTS_INSTRUCTIONS, DEFAULT_VOICE, RuntimeSettings, active_tools_for_profile, compose_system_prompt, list_profile_names, load_profile_character_prompt_by_name, load_profile_qwen_voice_by_name, load_profile_tts_instructions_by_name, load_profile_voice_by_name, normalize_profile_name, profile_has_prompt_audio_by_name, qwen_voice_choices, save_profile_definition, save_selected_profile_name, tsukasa_voice_choices


ROOT_DIR = Path(__file__).resolve().parents[2]
GUI_TOOL_NAMES = reachy_conversation_tools.GUI_TOOL_NAMES
AUDIO_LEVEL_DEFAULT = 50
AUDIO_CONTROL_TIMEOUT_S = 2.0


def _subprocess_stdout(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"command failed: {' '.join(command)}")
    return result.stdout.strip()


def _list_listening_pids(port: int) -> list[int]:
    output = _subprocess_stdout(["lsof", "-ti", f"tcp:{port}"])
    pids: list[int] = []
    for line in output.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return sorted(set(pids))


def _read_process_command(pid: int) -> str:
    return _subprocess_stdout(["ps", "-o", "command=", "-p", str(pid)])


def _is_same_conversation_app_process(pid: int, app_path: Path) -> bool:
    if pid == os.getpid():
        return False
    command = _read_process_command(pid)
    app_markers = {
        str(app_path),
        str(app_path.relative_to(ROOT_DIR)),
        "apps/conversation/main.py",
    }
    if not any(marker in command for marker in app_markers):
        return False
    return "uv" in command or ".venv/bin/python" in command


def _free_gradio_port_if_same_uv_app(port: int, app_path: Path) -> list[int]:
    killed_pids: list[int] = []
    for pid in _list_listening_pids(port):
        with suppress(Exception):
            if _is_same_conversation_app_process(pid, app_path):
                os.kill(pid, signal.SIGKILL)
                killed_pids.append(pid)
    return killed_pids


def _reachy_daemon_http_base_url(robot_host: str | None) -> str:
    host = str(robot_host or "").strip() or "reachy-mini.local"
    parsed = urlsplit(host if "://" in host else f"http://{host}")
    hostname = parsed.hostname or parsed.netloc or parsed.path
    if not hostname:
        raise ValueError("Reachy host is empty")
    scheme = parsed.scheme or "http"
    port = parsed.port or 8000
    return f"{scheme}://{hostname}:{port}"


def _coerce_audio_percent(value: SupportsFloat | str | None, *, fallback: int = AUDIO_LEVEL_DEFAULT) -> int:
    if value is None:
        return fallback
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return fallback


def _read_robot_audio_level(robot_host: str | None, endpoint: str, label: str) -> tuple[int, str]:
    response = httpx.get(
        f"{_reachy_daemon_http_base_url(robot_host)}/api/volume{endpoint}",
        timeout=AUDIO_CONTROL_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    percent = _coerce_audio_percent(payload.get("volume"))
    platform = str(payload.get("platform") or "unknown")
    device = str(payload.get("device") or "unknown device")
    return percent, f"{label}: {percent}% on {platform} / {device}"


def _write_robot_audio_level(robot_host: str | None, endpoint: str, label: str, percent: float) -> tuple[int, str]:
    requested = _coerce_audio_percent(percent)
    response = httpx.post(
        f"{_reachy_daemon_http_base_url(robot_host)}/api/volume{endpoint}",
        json={"volume": requested},
        timeout=AUDIO_CONTROL_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    applied = _coerce_audio_percent(payload.get("volume"), fallback=requested)
    platform = str(payload.get("platform") or "unknown")
    device = str(payload.get("device") or "unknown device")
    return applied, f"{label}: {applied}% on {platform} / {device}"


def _load_robot_audio_levels(robot_host: str | None) -> tuple[int, str, int, str]:
    try:
        volume, volume_status = _read_robot_audio_level(robot_host, "/current", "Volume Control")
    except Exception as exc:
        volume = AUDIO_LEVEL_DEFAULT
        volume_status = f"Volume Control unavailable: {exc}"

    try:
        microphone, microphone_status = _read_robot_audio_level(robot_host, "/microphone/current", "Microphone")
    except Exception as exc:
        microphone = AUDIO_LEVEL_DEFAULT
        microphone_status = f"Microphone unavailable: {exc}"

    return volume, volume_status, microphone, microphone_status


def build_gradio_ui(args: argparse.Namespace, runtime_settings: RuntimeSettings) -> gr.Blocks:
    profiles_dir = runtime_settings.profiles_dir
    profile_names = list_profile_names(profiles_dir)
    initial_profile, initial_tools, initial_character_prompt, _initial_prompt, initial_voice, initial_qwen_voice, initial_tts_instructions, _initial_version = runtime_settings.snapshot()
    initial_volume, _initial_volume_status, initial_microphone, _initial_microphone_status = _load_robot_audio_levels(args.robot_host)
    tsukasa_base_url = os.environ.get("TSUKASA_SPEECH_BASE_URL", "http://localhost:5001")
    qwen_dropdown_choices = qwen_voice_choices(include_custom=profile_has_prompt_audio_by_name(profiles_dir, initial_profile, "qwen3-tts"))

    def tsukasa_dropdown_update(profile: str, selected_voice: str):
        dropdown_choices = [
            (f"{name} - {description}", name)
            for name, description in tsukasa_voice_choices(
                selected_voice,
                base_url=tsukasa_base_url,
                include_custom=profile_has_prompt_audio_by_name(profiles_dir, profile, "tsukasa-speech"),
            )
        ]
        return gr.update(choices=dropdown_choices, value=selected_voice)

    def qwen_dropdown_update(profile: str, selected_voice: str):
        return gr.update(
            choices=qwen_voice_choices(include_custom=profile_has_prompt_audio_by_name(profiles_dir, profile, "qwen3-tts")),
            value=selected_voice,
        )

    def on_profile_change(profile: str):
        selected_profile = profile if profile in profile_names else initial_profile
        selected_tools = active_tools_for_profile(profiles_dir, selected_profile, GUI_TOOL_NAMES)
        character_prompt = load_profile_character_prompt_by_name(profiles_dir, selected_profile, DEFAULT_CHARACTER_PROMPT)
        voice = load_profile_voice_by_name(profiles_dir, selected_profile, DEFAULT_VOICE)
        qwen_voice = load_profile_qwen_voice_by_name(profiles_dir, selected_profile, DEFAULT_QWEN_VOICE)
        tts_instructions = load_profile_tts_instructions_by_name(profiles_dir, selected_profile, DEFAULT_TTS_INSTRUCTIONS)
        status = f"Loaded character '{selected_profile}'. You can edit the character prompt, Tsukasa voice, Qwen voice, save, or apply live."
        return character_prompt, selected_tools, tsukasa_dropdown_update(selected_profile, voice), qwen_dropdown_update(selected_profile, qwen_voice), tts_instructions, status

    def on_volume_change(volume: float) -> tuple[int, str]:
        try:
            return _write_robot_audio_level(args.robot_host, "/set", "Volume Control", volume)
        except Exception as exc:
            safe_volume = _coerce_audio_percent(volume)
            return safe_volume, f"Volume Control unavailable: {exc}"

    def on_microphone_change(volume: float) -> tuple[int, str]:
        try:
            return _write_robot_audio_level(args.robot_host, "/microphone/set", "Microphone", volume)
        except Exception as exc:
            safe_volume = _coerce_audio_percent(volume)
            return safe_volume, f"Microphone unavailable: {exc}"

    def on_apply(
        profile: str,
        character_prompt: str,
        selected_tools: list[str],
        voice: str,
        qwen_voice: str,
        tts_instructions: str,
    ) -> str:
        if profile not in profile_names:
            return f"Unknown character '{profile}'."
        active_profile, enabled_tools, _character_prompt, _instructions, active_voice, active_qwen_voice, active_tts_instructions, _version = runtime_settings.update(
            profile,
            selected_tools,
            character_prompt,
            compose_system_prompt(character_prompt),
            voice,
            qwen_voice,
            tts_instructions,
            greeting_reason="character update",
        )
        save_selected_profile_name(profiles_dir, active_profile)
        return (
            f"Live runtime updated: character={active_profile}, tsukasa_voice={active_voice}, qwen_voice={active_qwen_voice}, tts_instructions={active_tts_instructions[:48]!r}, "
            f"tools={', '.join(enabled_tools) if enabled_tools else 'none'}"
        )

    def on_save(profile: str, character_prompt: str, selected_tools: list[str], voice: str, qwen_voice: str, tts_instructions: str) -> str:
        if profile not in profile_names:
            return f"Unknown character '{profile}'. Create it first."
        profile_dir = save_profile_definition(profiles_dir, profile, character_prompt, selected_tools, voice, qwen_voice, tts_instructions, GUI_TOOL_NAMES)
        runtime_settings.update(
            profile,
            selected_tools,
            character_prompt,
            compose_system_prompt(character_prompt),
            voice,
            qwen_voice,
            tts_instructions,
            greeting_reason="character update",
        )
        save_selected_profile_name(profiles_dir, profile)
        return f"Saved character '{profile}' to {profile_dir}."

    def on_create(profile_name: str, character_prompt: str, selected_tools: list[str], voice: str, qwen_voice: str, tts_instructions: str):
        nonlocal profile_names
        normalized_name = normalize_profile_name(profile_name)
        if not normalized_name:
            return (
                gr.update(),
                character_prompt,
                selected_tools,
                voice,
                qwen_voice,
                tts_instructions,
                "Enter a valid character name.",
                profile_name,
            )

        profile_dir = save_profile_definition(
            profiles_dir,
            normalized_name,
            character_prompt or DEFAULT_CHARACTER_PROMPT,
            selected_tools or active_tools_for_profile(profiles_dir, initial_profile, GUI_TOOL_NAMES),
            voice or DEFAULT_VOICE,
            qwen_voice or initial_qwen_voice,
            tts_instructions or DEFAULT_TTS_INSTRUCTIONS,
            GUI_TOOL_NAMES,
        )
        profile_names = sorted(set([*profile_names, normalized_name]))
        saved_character_prompt = load_profile_character_prompt_by_name(profiles_dir, normalized_name, DEFAULT_CHARACTER_PROMPT)
        saved_tools = active_tools_for_profile(profiles_dir, normalized_name, GUI_TOOL_NAMES)
        saved_voice = load_profile_voice_by_name(profiles_dir, normalized_name, DEFAULT_VOICE)
        saved_qwen_voice = load_profile_qwen_voice_by_name(profiles_dir, normalized_name, DEFAULT_QWEN_VOICE)
        saved_tts_instructions = load_profile_tts_instructions_by_name(profiles_dir, normalized_name, DEFAULT_TTS_INSTRUCTIONS)
        runtime_settings.update(
            normalized_name,
            saved_tools,
            saved_character_prompt,
            compose_system_prompt(saved_character_prompt),
            saved_voice,
            saved_qwen_voice,
            saved_tts_instructions,
        )
        save_selected_profile_name(profiles_dir, normalized_name)
        return (
            gr.update(choices=profile_names, value=normalized_name),
            saved_character_prompt,
            saved_tools,
            tsukasa_dropdown_update(normalized_name, saved_voice),
            qwen_dropdown_update(normalized_name, saved_qwen_voice),
            saved_tts_instructions,
            f"Created character '{normalized_name}' at {profile_dir}.",
            "",
        )

    with gr.Blocks(title="Reachy Mini Conversation Settings") as demo:
        gr.Markdown(
            "# Reachy Mini Conversation\n"
            "Use this panel to switch character, edit only the character-specific prompt, choose Tsukasa and Qwen voices separately, and save while the local conversation loop is running."
        )
        with gr.Row():
            profile_dropdown = gr.Dropdown(label="Character", choices=profile_names, value=initial_profile)
            new_profile_box = gr.Textbox(label="New Character Name", placeholder="new_character", interactive=True)
            create_button = gr.Button("Create Character")
        character_box = gr.Textbox(label="Character Prompt", value=initial_character_prompt, lines=12, interactive=True)
        voice_dropdown = gr.Dropdown(
            label="Tsukasa Voice",
            choices=[
                (f"{name} - {description}", name)
                for name, description in tsukasa_voice_choices(
                    initial_voice,
                    base_url=tsukasa_base_url,
                    include_custom=profile_has_prompt_audio_by_name(profiles_dir, initial_profile, "tsukasa-speech"),
                )
            ],
            value=initial_voice,
            allow_custom_value=True,
        )
        qwen_voice_dropdown = gr.Dropdown(label="Qwen Voice", choices=qwen_dropdown_choices, value=initial_qwen_voice)
        tts_instructions_box = gr.Textbox(label="TTS Instructions", value=initial_tts_instructions, lines=4, interactive=True)
        tool_checkboxes = gr.CheckboxGroup(label="Enabled Tools", choices=GUI_TOOL_NAMES, value=initial_tools)
        status_box = gr.Textbox(label="Status", value="Ready.", interactive=False)
        with gr.Row():
            apply_button = gr.Button("Apply Live", variant="primary")
            save_button = gr.Button("Save Character")
        with gr.Row():
            volume_slider = gr.Slider(label="Volume Control", minimum=0, maximum=100, step=1, value=initial_volume, interactive=True)
            microphone_slider = gr.Slider(label="Microphone", minimum=0, maximum=100, step=1, value=initial_microphone, interactive=True)

        profile_dropdown.change(
            on_profile_change,
            inputs=[profile_dropdown],
            outputs=[character_box, tool_checkboxes, voice_dropdown, qwen_voice_dropdown, tts_instructions_box, status_box],
        )
        apply_button.click(
            on_apply,
            inputs=[profile_dropdown, character_box, tool_checkboxes, voice_dropdown, qwen_voice_dropdown, tts_instructions_box],
            outputs=[status_box],
        )
        save_button.click(
            on_save,
            inputs=[profile_dropdown, character_box, tool_checkboxes, voice_dropdown, qwen_voice_dropdown, tts_instructions_box],
            outputs=[status_box],
        )
        create_button.click(
            on_create,
            inputs=[new_profile_box, character_box, tool_checkboxes, voice_dropdown, qwen_voice_dropdown, tts_instructions_box],
            outputs=[profile_dropdown, character_box, tool_checkboxes, voice_dropdown, qwen_voice_dropdown, tts_instructions_box, status_box, new_profile_box],
        )
        volume_slider.input(
            on_volume_change,
            inputs=[volume_slider],
            outputs=[volume_slider, status_box],
        )
        microphone_slider.input(
            on_microphone_change,
            inputs=[microphone_slider],
            outputs=[microphone_slider, status_box],
        )
    return demo


def launch_gradio_ui(args: argparse.Namespace, runtime_settings: RuntimeSettings):
    demo = build_gradio_ui(args, runtime_settings)
    app_path = Path(__file__).resolve().parent / "main.py"
    killed_pids = _free_gradio_port_if_same_uv_app(args.gradio_port, app_path)
    if killed_pids:
        logging.getLogger("conversation.app").warning(
            "Killed existing conversation app process(es) on port %d: %s",
            args.gradio_port,
            ", ".join(str(pid) for pid in killed_pids),
        )
    logging.getLogger("conversation.app").info(
        "[bold]Starting Gradio GUI[/] http://%s:%d",
        args.gradio_host,
        args.gradio_port,
    )
    try:
        demo.launch(
            server_name=args.gradio_host,
            server_port=args.gradio_port,
            prevent_thread_lock=True,
            quiet=not args.debug,
            show_error=True,
            inbrowser=False,
        )
    except Exception:
        with suppress(Exception):
            demo.close()
        raise
    return demo