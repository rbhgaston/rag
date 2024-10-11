# FEATURE EXTRACTION
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import os 
import shutil
import numpy as np

DATA_PATH = "/Users/ghassenbenhamida/Documents/projects/rag/data/games"
BASE_CHROMA_PATH = "chroma"
CHUNK_SIZES = [256, 512, 1024]

def set_DATA_PATH(path):
    DATA_PATH = path

# EMBEDDINGS
class OllamaEmbeddingsNormalized(OllamaEmbeddings):
    # overrride the process_emb_response method to normalize the embeddings
    def _process_emb_response(self, input: str) -> list[float]:
        emb = super()._process_emb_response(input)
        return (np.array(emb) / np.linalg.norm(emb)).tolist()

EMBEDDINGS = OllamaEmbeddingsNormalized(
    model="mxbai-embed-large",
    model_kwargs={'device': 'cpu'},
    show_progress=True
    # multi_process=True,
    # encode_kwargs={'normalize_embeddings': True},
)


def calculate_chunk_ids(chunks, CHUNK_SIZE):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{CHUNK_SIZE}:{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return [x.metadata["id"] for x in chunks]   


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


def create_chroma(docs):
    # create 3 different db for each chunk size
    for CHUNK_SIZE in CHUNK_SIZES:
        CHROMA_PATH = f"{BASE_CHROMA_PATH}_{CHUNK_SIZE}"
        #remove db if exists
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        print(f"Creating ChromaDB in {CHROMA_PATH}")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDINGS)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=int(CHUNK_SIZE/10),
            length_function=len,
            add_start_index=True)

        chunks = text_splitter.split_documents(docs)

        db.add_documents(
            documents=chunks,
            ids=calculate_chunk_ids(chunks, CHUNK_SIZE),
        )


def create_db():
    docs = load_documents()
    create_chroma(docs)

    print("ChromaDB created")


if __name__ == "__main__":
    create_db()


