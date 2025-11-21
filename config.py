import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

FINBERT_MODEL_PATH = "./model_outputs/finbert_ner_20251105_161501/checkpoint-4820"


DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
