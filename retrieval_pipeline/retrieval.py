from asyncio import timeout
from ingestion_pipeline.embeddings import Embeddings, VectorStore
from langchain_groq import ChatGroq
import os
import dotenv

dotenv.load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY not found in the environment variables") 


class RetrievalPipeline:
    def __init__(self, vector_store: VectorStore, embeddings: Embeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):

        ## we generate embeddings for the query
        query_embeddings = self.embeddings.generate_embeddings([query])[0]

        ## we retrieve the documents from the vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings = [query_embeddings.tolist()],
                n_results = top_k,
                include = ["metadatas", "distances", "documents", "embeddings"],
            )
            
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
                return retrieved_docs
            
        except Exception as e:
            print(f"Error retrieving documents from vector store: {e}")


class LLMRetrieval:
    def __init__(self, model_name: str = "qwen/qwen3-32b"):
        self.model_name = model_name
        self.llm = ChatGroq(
            model_name = self.model_name,
            temperature = 0,
            timeout = None,
            max_retries = 2,
            reasoning_format="hidden" 
        )    

    def generate_response(self, query: str, retrieved_docs: list[dict]):

        try:
            messages = [
                {"role": "system", 
                "content":( 
                "You are a helpful HR assistant specialized in CV and Resume analysis."
                "Use the retrieved documents to answer the user query."
                "If the answer is not in the context, say you don't know.")
                },

                {"role": "user", 
                f"content":(
                    f"Context:\n{retrieved_docs}\n\n"
                    f"User query:\n{query}"
                )},
            ]
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating response: {e}")    
            return None
