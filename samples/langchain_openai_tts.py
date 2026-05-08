import local_tts
from config import load_entrypoint_env

load_entrypoint_env(local_tts)


if __name__ == "__main__":
    local_tts.main()