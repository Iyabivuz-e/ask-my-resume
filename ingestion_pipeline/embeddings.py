
import uuid
from sentence_transformers import SentenceTransformer
from chromadb import Client, PersistentClient
import os
from typing import List, Any
import numpy as np

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
    

class VectorStore:
    def __init__(self, collection_name: str = "cv_collection", persistent_store: str = "./data/vectorStore"):
        self.collection_name = collection_name
        self.persistent_store = persistent_store
        self.collection = None,
        self._initialize_collection()

    def _initialize_collection(self):
        try:
            os.makedirs(self.persistent_store, exist_ok=True)
            self.client = PersistentClient(path = self.persistent_store) ## Initialize the client
            self.collection = self.client.get_or_create_collection(name = self.collection_name) ## Initialize the collection
        except Exception as e:
            print(f"Error initializing collection: {e}")

    def add_embeddings_to_collection(self, documents: List[Any], embeddings: np.ndarray):

        if len(documents)!= len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")
        
        idx = []
        document_list = []
        embedding_list = []

        print("Adding embeddings to collection...")
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4()}_ {i}"
            idx.append(doc_id)
            document_list.append(doc.page_content)
            embedding_list.append(embedding)
        
        try:
            self.collection.upsert(
                documents = document_list,
                embeddings = embedding_list,
                ids = idx,
            )
            print("Embeddings added to collection successfully!")   
        except Exception as e:
            print(f"Error adding embeddings to collection: {e}")


        