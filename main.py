from ingestion_pipeline.data_ingestion import DataIngestion
from ingestion_pipeline.embeddings import Embeddings

def main():
    print("Hello from ask-my-cv!")

    data_ingestion = DataIngestion("./data/Resume1.pdf")
    chunks = data_ingestion.chunck_text()
    embeddings = Embeddings()
    embeddings.generate_embeddings(chunks)

    print("Data ingestion completed successfully!")


if __name__ == "__main__":
    main()
