from ingestion_pipeline.embeddings import Embeddings, VectorStore

class RetrievalPipeline:
    def __init__(self, vector_store: VectorStore, embeddings: Embeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        print(f"Retrieving documents based on the user query: {query}")

        ## we generate embeddings for the query
        query_embeddings = self.embeddings.generate_embeddings([query])[0]
        print(f"Query embeddings generated successfully!")

        ## we retrieve the documents from the vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings = [query_embeddings.tolist()],
                n_results = top_k,
                include = ["metadatas", "distances", "documents", "embeddings"],
            )
            print(f"Documents retrieved successfully!")
            
            ## We then process the retrieved documents and calculate the score

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, doc, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    score = 1.0 /(1.0 + float(distance))

                    if score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "document": doc,
                            "metadata": metadata,
                            "distance": distance,
                            "score": score,
                            "rank": i + 1,
                        })
                print(f"Documents retrieved successfully! {len(retrieved_docs)} documents retrieved.")
                print(retrieved_docs[0]["document"])
                return retrieved_docs
            
        except Exception as e:
            print(f"Error retrieving documents from vector store: {e}")


class LLMRetrieval:
    