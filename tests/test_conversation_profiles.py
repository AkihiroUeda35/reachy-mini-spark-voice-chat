from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from main import (
    DEFAULT_CHARACTER_PROMPT,
    DEFAULT_TTS_INSTRUCTIONS,
    DEFAULT_VOICE,
    build_tts_request_voice,
    load_profile_character_prompt_by_name,
    load_profile_qwen_voice_by_name,
    load_profile_prompt_by_name,
    load_profile_tts_instructions_by_name,
    load_profile_voice_by_name,
    save_profile_definition,
)
from state import compose_system_prompt, load_selected_profile_name, save_selected_profile_name


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
                "captain",
                "Speak with gentle enthusiasm.",
            )

            self.assertEqual(load_profile_voice_by_name(profiles_dir, "tester", DEFAULT_VOICE), "captain")
            self.assertEqual(load_profile_qwen_voice_by_name(profiles_dir, "tester"), "Ryan")
            self.assertEqual(
                build_tts_request_voice(
                    load_profile_voice_by_name(profiles_dir, "tester", DEFAULT_VOICE),
                    load_profile_qwen_voice_by_name(profiles_dir, "tester"),
                ),
                {"tsukasa-speech": "captain", "qwen3-tts": "Ryan"},
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
                "Speak with gentle enthusiasm.",
            )

            self.assertEqual(load_profile_qwen_voice_by_name(profiles_dir, "tester"), "Ryan")
            saved_payload = (tester_dir / "voice.json").read_text(encoding="utf-8")
            self.assertIn('"qwen_voice": "Ryan"', saved_payload)
            self.assertFalse((tester_dir / "voice.txt").exists())

    def test_selected_profile_defaults_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)
            save_profile_definition(
                profiles_dir,
                "tester",
                "## CHARACTER\n\nBe playful.",
                ["camera"],
                "Sohee",
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
                DEFAULT_TTS_INSTRUCTIONS,
            )

            save_selected_profile_name(profiles_dir, "tester")

            self.assertEqual(load_selected_profile_name(profiles_dir), "tester")

    def test_selected_profile_falls_back_when_saved_profile_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir)

            save_selected_profile_name(profiles_dir, "tester")

            self.assertEqual(load_selected_profile_name(profiles_dir), "default")


if __name__ == "__main__":
    unittest.main()