
from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
    
    def generate_embeddings(self, documents: list[str]):
        print("Embeddings generating start......!")
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts)
        print(f"Embeddings generated successfully: {len(embeddings)}")
        return embeddings
    
    def save_embeddings_to_vector_store(self):
        pass

    def save_metadata_to_vector_store(self):
        pass