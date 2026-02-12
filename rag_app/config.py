import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "mistral:latest"  
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"  
CHROMA_COLLECTION_NAME = "nobel_pprzes_info"
CHROMA_PERSIST_DIRECTORY = str(CHROMA_DB_DIR)
CSV_FILE_PATH = DATA_DIR / "nobel.csv"
TEXT_COLUMNS = [
    "category", "categoryFullName", "motivation", "categoryTopMotivation",
    "prizeAmount", "prizeAmountAdjusted", "dateAwarded", "awardYear",
    "fullName", "givenName", "familyName", "penName", "gender",
    "birth_date", "birth_city", "birth_country", "birth_continent",
    "death_date", "death_city", "death_country", "death_continent",
    "orgName", "nativeName", "acronym", "org_founded_date", 
    "org_founded_city", "org_founded_country", "org_founded_continent",
    "residence_1", "residence_2", "affiliation_1", "affiliation_2", 
    "affiliation_3", "affiliation_4",
    "ind_or_org"
]
METADATA_COLUMNS = [
    "id", "fullName", "orgName", "category", "awardYear", 
    "gender", "ind_or_org", "birth_country", "death_country", "dateAwarded"
]
RETRIEVAL_K = 5


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
