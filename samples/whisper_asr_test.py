import whisper_asr
from config import load_entrypoint_env

load_entrypoint_env(whisper_asr)


if __name__ == "__main__":
    raise SystemExit(whisper_asr.main())