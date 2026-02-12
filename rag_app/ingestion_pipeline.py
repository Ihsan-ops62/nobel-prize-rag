import os
import shutil
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from utils import load_csv_data, create_text_splitter
from tqdm import tqdm
import config

# Ingestion pipeline to process CSV, create embeddings, and store in Chroma DB
class IngestionPipeline:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )

        # Initialize text splitter
        self.text_splitter = create_text_splitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        self.vectorstore = None

    def clear_existing_db(self):
        """Safely clear the existing Chroma DB (Windows-safe)"""
        if os.path.exists(config.CHROMA_PERSIST_DIRECTORY):
            try:
                # Close vectorstore if it's open
                if self.vectorstore:
                    self.vectorstore.persist()
                    self.vectorstore = None
            except Exception as e:
                print(f" Warning closing vectorstore: {e}")

            # Attempt to remove folder
            try:
                shutil.rmtree(config.CHROMA_PERSIST_DIRECTORY)
                print(" Existing Chroma DB cleared")
            except PermissionError:
                print(" Could not delete Chroma DB. Make sure no process is using it (Streamlit, Python, Explorer).")
                return

        os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        print(" Chroma DB directory ready")
    
    def process_csv_row_by_row(self, csv_path: str):
        """Process CSV into Document chunks"""
        documents = load_csv_data(
            csv_path,
            text_columns=config.TEXT_COLUMNS,
            metadata_columns=config.METADATA_COLUMNS
        )

        all_chunks = []
        for doc in tqdm(documents, desc="Splitting documents into chunks"):
            chunks = self.text_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata = doc.metadata.copy()
                chunk.metadata["chunk_id"] = f"{doc.metadata.get('row_index', 0)}_{i}"
            all_chunks.extend(chunks)

        print(f" Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def create_and_store_embeddings(self, chunks):
        """Create embeddings in batches and store in Chroma DB"""
        if not chunks:
            raise ValueError("No chunks provided for embedding.")

        batch_size = getattr(config, "BATCH_SIZE", 100)
        for i in tqdm(range(0, len(chunks), batch_size), desc="Storing embeddings"):
            batch = chunks[i:i + batch_size]
            if i == 0:
                self.vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=str(config.CHROMA_PERSIST_DIRECTORY),
                    collection_name=config.CHROMA_COLLECTION_NAME
                )
            else:
                self.vectorstore.add_documents(batch)

        print(f" Successfully stored {len(chunks)} chunks in Chroma DB")
        return self.vectorstore

    def run(self, csv_path: str):
        """Run the complete ingestion pipeline"""
        if not os.path.exists(csv_path):
            print(f" CSV file not found: {csv_path}")
            return None

        print(" Running data ingestion pipeline...")
        self.clear_existing_db()
        chunks = self.process_csv_row_by_row(csv_path)
        if not chunks:
            print(" No chunks created from CSV")
            return None

        return self.create_and_store_embeddings(chunks)


def main():
    pipeline = IngestionPipeline()
    pipeline.run(str(config.CSV_FILE_PATH))


if __name__ == "__main__":
    main()
