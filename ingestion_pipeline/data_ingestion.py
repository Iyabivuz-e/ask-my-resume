
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataIngestion:
    def __init__(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 70):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = self.load_data()

    ## Load data
    def load_data(self):
        data = PyPDFLoader(self.pdf_path).load()
        return data
    
    ## Convert text to chunks
    def chunck_text(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators =["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(self.documents)
        return chunks
    
