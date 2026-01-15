from asyncio import timeout
from ingestion_pipeline.embeddings import Embeddings, VectorStore
from langchain_groq import ChatGroq
import os
import dotenv
from typing import List, Any

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
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        self.llm = ChatGroq(
            model_name = self.model_name,
            temperature = 0.0,
            timeout = None,
            max_retries = 2,
            reasoning_format="hidden" 
        )    

    def generate_response(self, query: str, retrieved_docs: list[dict]):

        try:
            messages = [
                {"role": "system", 
                "content":( 
                "You are ONLY a helpful HR assistant specialized in CV and Resume analysis."
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


class RetrievalWithCitations(LLMRetrieval):
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        super().__init__(model_name)

    def format_retrieved_docs(self, retrieved_docs: list[Any]):
        context_list = []
        refs = []

        for i, doc in enumerate(retrieved_docs, start=1):
            context_list.append(f"[DOC {i}]\n{doc['document']}\n")
            refs.append(
                f"DOC {i}:\n"
                f"doc_id: {doc['id']}\n"
                f"Source: {doc['metadata']['source']}\n" 
                f"Page: {doc['metadata']['page_label']}\n"
                f"Score: {doc['score']}\n"
            )

        context = "\n".join(context_list)
        reference_for_llm = "\n".join(refs)
        # print(f"Context: {context}")
        # print(f"References: {references}")

        return context, reference_for_llm


    def generate_response(self, query: str, retrieved_docs: list[dict]):

        try:
            context, reference_for_llm = self.format_retrieved_docs(retrieved_docs)
            messages = [
                {"role": "system", 
                "content": 
                """You are a strict question-answering system.
                Rules:
                - Answer ONLY the question asked.
                - If the answer is not explicitly stated in the context, reply with: "I don't know."
                - Do not summarize.
                - Do not add extra information.
                - Quote the exact answer when possible.
                - Cite the document like [DOC X].
                """
                },

                {"role": "user", "content": f"""
                        Use ONLY the CONTEXT to answer the question.
                        Cite sources using [DOC X].
                        If the answer is not explicitly present, say: I don't know.

                        === CONTEXT ===
                        {context}

                        === REFERENCES ===
                        {reference_for_llm}

                        Question: {query}
                        """}

            ]
            
            response = self.llm.invoke(messages)
            return {
                "answer": response.content,
                "sources": retrieved_docs
            }
        except Exception as e:
            print(f"Error generating response: {e}")    
            return None