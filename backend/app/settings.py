from pathlib import Path
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
VECTORDB_DIR = BASE_DIR / "vectordb"


UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)


class AppConfig(BaseModel):
    embed_model: str = "mxbai-embed-large"
    chat_model: str = "llama3.2"
    collection_prefix: str = "collection_" # for folder naming


CONFIG = AppConfig()