from ingestion_pipeline.data_ingestion import DataIngestion
from ingestion_pipeline.embeddings import Embeddings, VectorStore

def ingest():
    data_ingestion = DataIngestion("./data/Resume1.pdf")
    chunks = data_ingestion.chunck_text()

    myembeddings = Embeddings()
    embeddings_list = myembeddings.generate_embeddings(chunks)

    vector_store = VectorStore()
    vector_store.add_embeddings_to_collection(chunks, embeddings_list)

    print("Data ingestion completed successfully!")

if __name__ == "__main__":
    ingest()
    