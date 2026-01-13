from ingestion_pipeline.data_ingestion import DataIngestion
from ingestion_pipeline.embeddings import Embeddings, VectorStore
from retrieval_pipeline.retrieval import RetrievalPipeline, LLMRetrieval


def main():
    
    user_query = "What is my name?"

    data_ingestion = DataIngestion("./data/Resume1.pdf")
    chunks = data_ingestion.chunck_text()

    myembeddings = Embeddings()
    embeddings_list = myembeddings.generate_embeddings(chunks)

    vector_store = VectorStore()
    vector_store.add_embeddings_to_collection(chunks, embeddings_list)

    retrieval = RetrievalPipeline(vector_store, myembeddings)
    retrieved_docs = retrieval.retrieve(user_query)

    llm_retrieval = LLMRetrieval()
    llm_retrieval.generate_response(user_query, retrieved_docs)


    print("Data ingestion completed successfully!")


if __name__ == "__main__":
    main()
