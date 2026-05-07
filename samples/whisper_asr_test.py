from pathlib import Path
import importlib
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

importlib.import_module("lib.config").load_env()
main = importlib.import_module("lib.whisper_asr").main


if __name__ == "__main__":
    raise SystemExit(main())