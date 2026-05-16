from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _load_tsukasa_api_module():
    module_name = "test_tsukasa_api"
    module_path = Path(__file__).resolve().parents[1] / "tts" / "tsukasa_speech" / "api.py"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for tts/tsukasa_speech/api.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


TSUKASA_API = _load_tsukasa_api_module()


class _FakeImportableModule:
    def __init__(self) -> None:
        self.model = object()
        self.device = "cpu"
        self.voice_style = np.array([[10.0, 20.0]], dtype=np.float32)
        self.prompt_style = np.array([[2.0, 4.0]], dtype=np.float32)
        self.last_ref_s = None

    def compute_style_through_clip(self, path: str) -> np.ndarray:
        return self.voice_style

    def Kotodama_Prompter(self, model: object, text: str, device: str) -> np.ndarray:
        return self.prompt_style

    def inference(self, text: str, ref_s: np.ndarray, **_: object) -> np.ndarray:
        self.last_ref_s = ref_s
        return np.zeros(32, dtype=np.float32)

    def trim_long_silences(self, audio: np.ndarray) -> np.ndarray:
        return audio


class TsukasaApiTests(unittest.TestCase):
    def test_synthesize_blends_voice_and_prompt_style_when_instructions_present(self) -> None:
        fake_importable = _FakeImportableModule()

        with patch.dict(os.environ, {"TSUKASA_SPEECH_PROMPT_STYLE_BLEND": "0.2"}, clear=False), patch.object(
            TSUKASA_API,
            "_load_runtime",
            return_value={"importable": fake_importable, "smart_phonemize": lambda text: text},
        ), patch.object(TSUKASA_API, "_resolve_voice_path", return_value=Path("/tmp/prompt.wav")), patch.object(
            TSUKASA_API,
            "_wav_bytes",
            return_value=b"wav",
        ):
            result = TSUKASA_API._synthesize_sync(
                TSUKASA_API.SynthesizeRequest(
                    text="テストです",
                    voice="data:audio/wav;base64,AAAA",
                    instructions="低く落ち着いた男性の声で話してください。",
                )
            )

        expected = (0.8 * fake_importable.voice_style) + (0.2 * fake_importable.prompt_style)
        np.testing.assert_allclose(fake_importable.last_ref_s, expected)
        self.assertEqual(result, b"wav")

    def test_synthesize_uses_voice_style_without_instructions(self) -> None:
        fake_importable = _FakeImportableModule()

        with patch.object(
            TSUKASA_API,
            "_load_runtime",
            return_value={"importable": fake_importable, "smart_phonemize": lambda text: text},
        ), patch.object(TSUKASA_API, "_resolve_voice_path", return_value=Path("/tmp/prompt.wav")), patch.object(
            TSUKASA_API,
            "_wav_bytes",
            return_value=b"wav",
        ):
            TSUKASA_API._synthesize_sync(
                TSUKASA_API.SynthesizeRequest(
                    text="テストです",
                    voice="data:audio/wav;base64,AAAA",
                )
            )

        np.testing.assert_allclose(fake_importable.last_ref_s, fake_importable.voice_style)


if __name__ == "__main__":
    unittest.main()