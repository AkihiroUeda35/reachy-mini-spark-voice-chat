from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from main import (
    DEFAULT_CHARACTER_PROMPT,
    DEFAULT_TTS_INSTRUCTIONS,
    DEFAULT_VOICE,
    _apply_profile_tts_prompt_assets,
    build_tts_request_voice,
    load_profile_character_prompt_by_name,
    load_profile_qwen_voice_by_name,
    load_profile_prompt_by_name,
    load_profile_tts_instructions_by_name,
    load_profile_tts_ref_audio_by_name,
    load_profile_tts_ref_text_by_name,
    load_profile_voice_by_name,
    save_profile_definition,
)
from state import DEFAULT_QWEN_VOICE, CUSTOM_VOICE, RuntimeSettings, compose_system_prompt, load_selected_profile_name, qwen_voice_choices, save_selected_profile_name, tsukasa_voice_choices


class ConversationProfileTests(unittest.TestCase):
    def test_compose_system_prompt_includes_common_and_character(self) -> None:
        prompt = compose_system_prompt("## CHARACTER\n\nBe direct.")
        self.assertIn("## IDENTITY", prompt)
        self.assertIn("## CHARACTER", prompt)
        self.assertIn("Be direct.", prompt)

    def test_save_and_load_profile_definition_persists_character_tools_and_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            save_profile_definition(
                profiles_dir,
                "tester",
                "## CHARACTER\n\nBe playful.",
                ["move_head", "camera"],
                "Sohee",
                None,
                "Speak with gentle enthusiasm.",
            )

            self.assertEqual(
                load_profile_character_prompt_by_name(profiles_dir, "tester", DEFAULT_CHARACTER_PROMPT),
                "## CHARACTER\n\nBe playful.",
            )
            self.assertEqual(load_profile_voice_by_name(profiles_dir, "tester", DEFAULT_VOICE), "Sohee")
            self.assertEqual(load_profile_qwen_voice_by_name(profiles_dir, "tester"), "Sohee")
            self.assertEqual(
                load_profile_tts_instructions_by_name(profiles_dir, "tester", DEFAULT_TTS_INSTRUCTIONS),
                "Speak with gentle enthusiasm.",
            )
            prompt = load_profile_prompt_by_name(profiles_dir, "tester", DEFAULT_CHARACTER_PROMPT)
            self.assertIn("## IDENTITY", prompt)
            self.assertIn("Be playful.", prompt)

    def test_save_preserves_qwen_voice_txt_when_profile_voice_is_tsukasa(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "voice.txt").write_text("Ryan\n", encoding="utf-8")

            save_profile_definition(
                profiles_dir,
                "tester",
                "## CHARACTER\n\nBe playful.",
                ["move_head", "camera"],
                "audio_ref",
                None,
                "Speak with gentle enthusiasm.",
            )

            self.assertEqual(load_profile_voice_by_name(profiles_dir, "tester", DEFAULT_VOICE), "audio_ref")
            self.assertEqual(load_profile_qwen_voice_by_name(profiles_dir, "tester"), "Ryan")
            self.assertEqual(
                build_tts_request_voice(
                    load_profile_voice_by_name(profiles_dir, "tester", DEFAULT_VOICE),
                    load_profile_qwen_voice_by_name(profiles_dir, "tester"),
                ),
                {"tsukasa-speech": "audio_ref", "qwen3-tts": "Ryan"},
            )

    def test_save_persists_qwen_voice_in_voice_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "voice.json").write_text(
                '{\n  "voice": "captain",\n  "qwen_voice": "Ryan",\n  "tts_instructions": "Speak with gentle enthusiasm."\n}\n',
                encoding="utf-8",
            )

            save_profile_definition(
                profiles_dir,
                "tester",
                "## CHARACTER\n\nBe playful.",
                ["move_head", "camera"],
                "captain",
                None,
                "Speak with gentle enthusiasm.",
            )

            self.assertEqual(load_profile_qwen_voice_by_name(profiles_dir, "tester"), "Ryan")
            saved_payload = (tester_dir / "voice.json").read_text(encoding="utf-8")
            self.assertIn('"qwen_voice": "Ryan"', saved_payload)
            self.assertFalse((tester_dir / "voice.txt").exists())

    def test_profile_prompt_assets_are_loaded_per_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "character.txt").write_text("## CHARACTER\n\nBe playful.\n", encoding="utf-8")
            (tester_dir / "japanese.wav").write_bytes(b"RIFFjp")
            (tester_dir / "english.wav").write_bytes(b"RIFFen")
            (tester_dir / "english.txt").write_text("Good evening, sir.", encoding="utf-8")

            ref_audio = load_profile_tts_ref_audio_by_name(profiles_dir, "tester")
            ref_text = load_profile_tts_ref_text_by_name(profiles_dir, "tester")

            self.assertIsInstance(ref_audio, dict)
            self.assertEqual(set(ref_audio.keys()), {"tsukasa-speech", "qwen3-tts"})
            self.assertTrue(ref_audio["tsukasa-speech"].startswith("data:audio/"))
            self.assertTrue(ref_audio["qwen3-tts"].startswith("data:audio/"))
            self.assertEqual(ref_text, {"qwen3-tts": "Good evening, sir."})

    def test_profile_prompt_assets_are_loaded_from_mp3_per_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "character.txt").write_text("## CHARACTER\n\nBe playful.\n", encoding="utf-8")
            (tester_dir / "japanese.mp3").write_bytes(b"ID3jp")
            (tester_dir / "english.mp3").write_bytes(b"ID3en")
            (tester_dir / "english.txt").write_text("Good evening, sir.", encoding="utf-8")

            ref_audio = load_profile_tts_ref_audio_by_name(profiles_dir, "tester")
            ref_text = load_profile_tts_ref_text_by_name(profiles_dir, "tester")

            self.assertIsInstance(ref_audio, dict)
            self.assertEqual(set(ref_audio.keys()), {"tsukasa-speech", "qwen3-tts"})
            self.assertTrue(ref_audio["tsukasa-speech"].startswith("data:audio/mpeg;base64,"))
            self.assertTrue(ref_audio["qwen3-tts"].startswith("data:audio/mpeg;base64,"))
            self.assertEqual(ref_text, {"qwen3-tts": "Good evening, sir."})

    def test_custom_voice_selection_loads_profile_prompt_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "japanese.wav").write_bytes(b"RIFFjp")
            (tester_dir / "english.wav").write_bytes(b"RIFFen")
            (tester_dir / "english.txt").write_text("Good evening, sir.", encoding="utf-8")

            args = Mock()
            args.profile_voice_choice = CUSTOM_VOICE
            args.profile_qwen_voice_choice = CUSTOM_VOICE

            _apply_profile_tts_prompt_assets(args, profiles_dir, "tester")

            self.assertEqual(set(args.tts_ref_audio.keys()), {"tsukasa-speech", "qwen3-tts"})
            self.assertEqual(args.tts_ref_text, {"qwen3-tts": "Good evening, sir."})
            self.assertFalse(args.tts_x_vector_only_mode)

    def test_preset_voice_selection_skips_profile_prompt_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "japanese.wav").write_bytes(b"RIFFjp")
            (tester_dir / "english.wav").write_bytes(b"RIFFen")
            (tester_dir / "english.txt").write_text("Good evening, sir.", encoding="utf-8")

            args = Mock()
            args.profile_voice_choice = "audio_ref"
            args.profile_qwen_voice_choice = "Ryan"

            _apply_profile_tts_prompt_assets(args, profiles_dir, "tester")

            self.assertIsNone(args.tts_ref_audio)
            self.assertIsNone(args.tts_ref_text)
            self.assertFalse(args.tts_x_vector_only_mode)

    def test_custom_voice_choice_falls_back_when_prompt_audio_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "voice.json").write_text(
                '{\n  "voice": "custom",\n  "qwen_voice": "custom",\n  "tts_instructions": "Speak with gentle enthusiasm."\n}\n',
                encoding="utf-8",
            )

            self.assertEqual(load_profile_voice_by_name(profiles_dir, "tester", DEFAULT_VOICE), DEFAULT_VOICE)
            self.assertEqual(load_profile_qwen_voice_by_name(profiles_dir, "tester", DEFAULT_QWEN_VOICE), DEFAULT_QWEN_VOICE)

    def test_custom_voice_request_uses_backend_fallback_voices(self) -> None:
        self.assertEqual(
            build_tts_request_voice(CUSTOM_VOICE, CUSTOM_VOICE),
            {"tsukasa-speech": DEFAULT_VOICE, "qwen3-tts": DEFAULT_QWEN_VOICE},
        )

    def test_runtime_settings_preserve_custom_qwen_voice_when_prompt_audio_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "english.wav").write_bytes(b"RIFFen")

            settings = RuntimeSettings(
                profiles_dir=profiles_dir,
                active_profile="tester",
                enabled_tools=[],
                active_character_prompt="",
                active_instructions="",
                active_voice=DEFAULT_VOICE,
                active_qwen_voice=DEFAULT_QWEN_VOICE,
                active_tts_instructions="",
                gui_tool_names=[],
            )

            _profile, _tools, _character, _instructions, _voice, qwen_voice, _tts_instructions, _version = settings.update(
                "tester",
                [],
                "character",
                "instructions",
                DEFAULT_VOICE,
                CUSTOM_VOICE,
                "tts instructions",
            )

            self.assertEqual(qwen_voice, CUSTOM_VOICE)

    def test_voice_choices_include_custom_when_profile_prompt_audio_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "japanese.wav").write_bytes(b"RIFFjp")
            (tester_dir / "english.wav").write_bytes(b"RIFFen")

            self.assertEqual(tsukasa_voice_choices(base_url="", include_custom=True)[0][0], CUSTOM_VOICE)
            self.assertEqual(qwen_voice_choices(include_custom=True)[0], CUSTOM_VOICE)

    def test_missing_english_txt_is_auto_generated_from_english_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "character.txt").write_text("## CHARACTER\n\nBe playful.\n", encoding="utf-8")
            (tester_dir / "english.wav").write_bytes(b"RIFFen")

            with patch("state._transcribe_profile_prompt_audio", return_value="Good evening, sir."):
                ref_text = load_profile_tts_ref_text_by_name(profiles_dir, "tester")

            self.assertEqual(ref_text, {"qwen3-tts": "Good evening, sir."})
            self.assertEqual((tester_dir / "english.txt").read_text(encoding="utf-8"), "Good evening, sir.\n")

    def test_save_auto_generates_missing_english_txt_when_english_wav_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "english.wav").write_bytes(b"RIFFen")

            with patch("state._transcribe_profile_prompt_audio", return_value="Good evening, sir."):
                save_profile_definition(
                    profiles_dir,
                    "tester",
                    "## CHARACTER\n\nBe playful.",
                    ["camera"],
                    "audio_ref",
                    "Ryan",
                    DEFAULT_TTS_INSTRUCTIONS,
                )

            self.assertEqual((tester_dir / "english.txt").read_text(encoding="utf-8"), "Good evening, sir.\n")

    def test_save_auto_generates_missing_english_txt_when_english_mp3_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            tester_dir = profiles_dir / "tester"
            tester_dir.mkdir(parents=True, exist_ok=True)
            (tester_dir / "english.mp3").write_bytes(b"ID3en")

            with patch("state._transcribe_profile_prompt_audio", return_value="Good evening, sir."):
                save_profile_definition(
                    profiles_dir,
                    "tester",
                    "## CHARACTER\n\nBe playful.",
                    ["camera"],
                    "audio_ref",
                    "Ryan",
                    DEFAULT_TTS_INSTRUCTIONS,
                )

            self.assertEqual((tester_dir / "english.txt").read_text(encoding="utf-8"), "Good evening, sir.\n")

    def test_selected_profile_defaults_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            save_profile_definition(
                profiles_dir,
                "tester",
                "## CHARACTER\n\nBe playful.",
                ["camera"],
                "Sohee",
                None,
                DEFAULT_TTS_INSTRUCTIONS,
            )

            self.assertEqual(load_selected_profile_name(profiles_dir), "default")

    def test_selected_profile_is_loaded_when_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            save_profile_definition(
                profiles_dir,
                "tester",
                "## CHARACTER\n\nBe playful.",
                ["camera"],
                "Sohee",
                None,
                DEFAULT_TTS_INSTRUCTIONS,
            )

    def test_save_persists_explicit_qwen_voice_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)

            save_profile_definition(
                profiles_dir,
                "tester",
                "## CHARACTER\n\nBe playful.",
                ["camera"],
                "samurai",
                "Ryan",
                DEFAULT_TTS_INSTRUCTIONS,
            )

            self.assertEqual(load_profile_voice_by_name(profiles_dir, "tester", DEFAULT_VOICE), "samurai")
            self.assertEqual(load_profile_qwen_voice_by_name(profiles_dir, "tester"), "Ryan")

            save_selected_profile_name(profiles_dir, "tester")

            self.assertEqual(load_selected_profile_name(profiles_dir), "tester")

    def test_selected_profile_falls_back_when_saved_profile_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)

            save_selected_profile_name(profiles_dir, "tester")

            self.assertEqual(load_selected_profile_name(profiles_dir), "default")


if __name__ == "__main__":
    unittest.main()