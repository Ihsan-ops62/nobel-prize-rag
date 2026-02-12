import sys
import os

from ingestion_pipeline import main as ingestion_main
from retrieval_pipeline import RetrievalPipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py ingest       - Run data ingestion")
        print("  python main.py chat         - Run interactive chat (CLI)")
        print("  streamlit run streamlit_app.py  - Run web UI")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        print(" Running data ingestion pipeline...")
        ingestion_main()
    elif command == "chat":
        print(" Interactive chat mode is not implemented in this pipeline version.")
        print("Please use the Streamlit UI:")
        print("  streamlit run streamlit_app.py")
        sys.exit(0)
    else:
        print(f" Unknown command: {command}")
        print("Use 'ingest' or 'chat'.")
