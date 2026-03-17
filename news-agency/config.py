import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEWSROOM_MODEL = os.getenv("NEWSROOM_MODEL", "gpt-4o-mini")
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
MIN_SOURCES = int(os.getenv("MIN_SOURCES", "3"))