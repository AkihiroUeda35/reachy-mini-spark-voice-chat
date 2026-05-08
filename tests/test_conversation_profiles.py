from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from main import (
    DEFAULT_CHARACTER_PROMPT,
    DEFAULT_VOICE,
    load_profile_character_prompt_by_name,
    load_profile_prompt_by_name,
    load_profile_voice_by_name,
    save_profile_definition,
)
from state import compose_system_prompt


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
            )

            self.assertEqual(
                load_profile_character_prompt_by_name(profiles_dir, "tester", DEFAULT_CHARACTER_PROMPT),
                "## CHARACTER\n\nBe playful.",
            )
            self.assertEqual(load_profile_voice_by_name(profiles_dir, "tester", DEFAULT_VOICE), "Sohee")
            prompt = load_profile_prompt_by_name(profiles_dir, "tester", DEFAULT_CHARACTER_PROMPT)
            self.assertIn("## IDENTITY", prompt)
            self.assertIn("Be playful.", prompt)


if __name__ == "__main__":
    unittest.main()