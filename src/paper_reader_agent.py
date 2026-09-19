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


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


pdf_path = "data/1706.03762v7.pdf"


def load_and_chunk_pdf(pdf_path: str):
    """
    Loads a PDF and splits it into smaller chunks.
    """

    print(f"Loading PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(pages)

    # Store metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    print(f"Split into {len(chunks)} chunks")

    return chunks


def create_vector_store(chunks, collection_name="paper_collection"):
    """
    Converts chunks into embeddings and stores them in ChromaDB.
    """

    print("Creating embeddings and storing in ChromaDB")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )

    print("Vector Store Created")

    return vector_store

def summarize_chunks(chunks, batch_size=8):
    """
    Summarizes the PDF in smaller batches.

    Instead of sending the entire PDF to Gemini
    in one huge request, the chunks are processed
    in smaller groups.
    """

    batch_summaries = []

    for i in range(0, len(chunks), batch_size):

        # Get the current batch
        batch = chunks[i:i + batch_size]

        context = ""

        for j, doc in enumerate(batch):

            page_number = doc.metadata.get("page", "Unknown")

            context += f"""
[Page {page_number}]
{doc.page_content}
"""

        start_chunk = i + 1
        end_chunk = min(i + batch_size, len(chunks))

        print(
            f"Summarizing chunks {start_chunk} "
            f"to {end_chunk}..."
        )

        messages = [

            SystemMessage(
                content="""You are a research paper analyst.

Summarize ONLY the provided section of the research paper.

Extract the important information while preserving
the meaning of the original paper.

Focus on information such as:

- Research problem and motivation
- Main objective
- Methodology or approach
- Important techniques or components
- Experiments or results
- Findings
- Limitations
- Conclusions

Do not invent information that is not present
in the provided text."""
            ),

            HumanMessage(
                content=f"""
Paper section:

{context}

Provide a concise but information-rich summary
of this section.
"""
            )
        ]

        response = llm.invoke(messages)

        batch_summaries.append(response.content)

    return batch_summaries


def paper_reader_agent(pdf_path: str, question: str) -> dict:
    """
    Reads a PDF and answers a question about it
    using RAG.

    For summaries:
        Uses batch summarization without embeddings.

    For normal questions:
        Uses Chroma similarity search.
    """

    chunks = load_and_chunk_pdf(pdf_path)

    print("Searching...")


    if "summar" in question.lower():

        print(
            f"Summarization request detected. "
            f"Using all {len(chunks)} chunks."
        )

        batch_summaries = summarize_chunks(
            chunks,
            batch_size=8
        )

        print(
            f"\nCreated {len(batch_summaries)} "
            f"section summaries."
        )

        combined_summary = ""

        for i, summary in enumerate(batch_summaries):

            combined_summary += f"""

========== SECTION SUMMARY {i + 1} ==========

{summary}
"""

        print("Creating final summary...")

        messages = [

            SystemMessage(
                content="""You are a research paper analyst.

You are given summaries of different sections
of the same research paper.

Using ONLY these summaries, create a coherent
and comprehensive summary of the entire paper.

Cover the following where the information is available:

- Research problem and motivation
- Main objective
- Proposed approach or methodology
- Important techniques or components
- Key results or findings
- Limitations
- Conclusion
- Overall contribution

Do not add information that is not supported
by the provided summaries.

Do not mention that the paper was summarized
in batches.

Organize the final answer clearly with headings
and concise explanations."""
            ),

            HumanMessage(
                content=f"""
Section summaries from the research paper:

{combined_summary}

Question:

{question}

Now provide a comprehensive summary of the
entire research paper.
"""
            )
        ]

        response = llm.invoke(messages)

        return {
            "question": question,
            "answer": response.content,
            "chunks_used": len(chunks)
        }

    else:

        vector_store = create_vector_store(chunks)

        relevant_docs = vector_store.similarity_search(
            question,
            k=4
        )

        print(
            f"Retrieved {len(relevant_docs)} "
            f"relevant chunks."
        )


        context = ""

        for i, doc in enumerate(relevant_docs):

            page_number = doc.metadata.get(
                "page",
                "Unknown"
            )

            context += f"""
[Chunk {i + 1} | Page {page_number}]
{doc.page_content}
"""

        messages = [

            SystemMessage(
                content="""You are a research paper analyst.

Use ONLY the provided context from the paper.

Answer the user's question directly using
the relevant information from the context.

Do not add information that is not present
in the paper.

Be precise and clearly distinguish between
different sections or findings of the paper."""
            ),

            HumanMessage(
                content=f"""
Context from paper:

{context}

Question:

{question}
"""
            )
        ]

        response = llm.invoke(messages)

        return {
            "question": question,
            "answer": response.content,
            "chunks_used": len(relevant_docs)
        }


if __name__ == "__main__":

    pdf_path = "data/1706.03762v7.pdf"

    question = "Summarize the uploaded PDF"

    result = paper_reader_agent(
        pdf_path,
        question
    )

    print("\n" + "=" * 50)

    print(
        f"QUESTION: {result['question']}"
    )

    print("=" * 50)

    print(
        f"\nANSWER:\n{result['answer']}"
    )

    print(
        f"\n(Used {result['chunks_used']} chunks)"
    )