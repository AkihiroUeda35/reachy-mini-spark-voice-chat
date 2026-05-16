from __future__ import annotations

import sys
import tempfile
import types
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import whisper_asr


class _FakeAudioSegment:
    channels = 1
    frame_rate = 24000
    sample_width = 2

    @classmethod
    def from_file(cls, _path: str):
        return cls()

    def get_array_of_samples(self):
        return array("h", [0, 16384, -16384, 8192])


class WhisperAsrTests(unittest.TestCase):
    def test_prepared_audio_from_file_falls_back_to_pydub_for_mp3(self) -> None:
        fake_pydub = types.ModuleType("pydub")
        setattr(fake_pydub, "AudioSegment", _FakeAudioSegment)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "english.mp3"
            audio_path.write_bytes(b"ID3test")

            with patch.object(whisper_asr.sf, "read", side_effect=RuntimeError("unsupported format")), patch.dict(
                sys.modules,
                {"pydub": fake_pydub},
            ):
                prepared = whisper_asr._prepared_audio_from_file(audio_path, 16000)

        self.assertEqual(prepared.filename, "english.mp3")
        self.assertEqual(prepared.mime_type, "audio/mpeg")
        self.assertEqual(prepared.sample_rate, 16000)
        self.assertTrue(prepared.pcm16_bytes)


if __name__ == "__main__":
    unittest.main()