from ingestion_pipeline.embeddings import Embeddings, VectorStore
from retrieval_pipeline.retrieval import RetrievalPipeline, LLMRetrieval


def main():

    print("Starting the chatbot...")
    
    vector_store = VectorStore()
    myembeddings = Embeddings()

    retrieval = RetrievalPipeline(vector_store, myembeddings)
    llm_retrieval = LLMRetrieval()

    ## Interaction loop
    while True:
        user_query = input("\nAsk (type 'q' to quit): ")
        if user_query.lower() in "q":
            break
        
        retrieved_docs = retrieval.retrieve(user_query)
        response = llm_retrieval.generate_response(user_query, retrieved_docs)
        print("\nAssistant:", response)


if __name__ == "__main__":
    main()
