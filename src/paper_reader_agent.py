import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

# initialise model
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
    temperature = 0.3
)

# initilaise gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

pdf_path = "data/1706.03762v7.pdf"
def load_and_chunk_pdf(pdf_path : str):
    """
    Loads a pdf and splits it into smaller chunks for embedding.
    """
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_documents(pages)

    # store metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks, collection_name="paper_collection"):
    """
    Converts chunks into embeddings and stores them in ChromaDB.
    """
    print(f"Creating embeddings and storing in ChromaDB")
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        collection_name = collection_name,
        persist_directory = "./chroma_db"
    )
    print("Vector Store Created")
    return vector_store

def paper_reader_agent(pdf_path: str, question: str) -> dict:
    """
    Reads a PDF and answers a question about it using RAG.
    """

    # load and chunk 
    chunks = load_and_chunk_pdf(pdf_path)

    # create vector store
    vector_store = create_vector_store(chunks)

    # find relevant chunks
    print(f"Searching...")
    relevant_docs = vector_store.similarity_search(question, k=4)

    context = ""
    for i, doc in enumerate(relevant_docs):
        page_number = doc.metadata.get("page", "Unknown")
        context += f"""
    [Chunk {i+1} | Page {page_number}]
    {doc.page_content}
    """

    # use gemini to answer only the relevant context
    messages = [
        SystemMessage(content="""You are a research paper analyst. Answer questions using ONLY 
        the provided context from the paper. If the answer isn't in the context, say so clearly. 
        Be precise and cite specific details from the text."""),
        HumanMessage(content=f"Context from paper:\n{context}\n\nQuestion: {question}")
    ]

    response = llm.invoke(messages)

    return{
        "question" : question,
        "answer" : response.content,
        "chunks_used" : len(relevant_docs)
    }

# Test the agent
if __name__ == "__main__":

    pdf_path = "data/1706.03762v7.pdf"
    question = "What is the main contribution of this paper?"
    
    result = paper_reader_agent(pdf_path, question)
    
    print("\n" + "="*50)
    print(f"QUESTION: {result['question']}")
    print("="*50)
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\n(Used {result['chunks_used']} relevant chunks)")