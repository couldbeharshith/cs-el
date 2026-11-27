from pathlib import Path

from dotenv import load_dotenv


ROOT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_root_env() -> None:
    load_dotenv(ROOT_ENV_FILE)
