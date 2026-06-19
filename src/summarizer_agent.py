import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.4
)

def summarizer_agent(content: str, source_label: str = "Unknown Source") -> dict:
    """
    Condenses raw text into clean, structured key points.
    """
    messages = [
        SystemMessage(content="""You are a precision summarizer. Your job:
        1. Extract only the most important points from the given content
        2. Write in clear, concise bullet points (max 5-7 points)
        3. Preserve technical accuracy - do not oversimplify to the point of being wrong
        4. Do not add any information not present in the original content
        5. If the content is too short or unclear to summarize meaningfully, say so"""),
        HumanMessage(content=f"Source: {source_label}\n\nContent to summarize:\n{content}")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "source": source_label,
        "summary": response.content
    }


# Test the agent
if __name__ == "__main__":
    sample_content = """
    Retrieval Augmented Generation (RAG) combines the parametric knowledge of large 
    language models with non-parametric knowledge retrieved from external sources. 
    This approach addresses key limitations of standalone LLMs, including hallucination, 
    outdated knowledge, and lack of source attribution. RAG systems typically consist of 
    a retriever component, which finds relevant documents from a knowledge base, and a 
    generator component, which produces responses conditioned on the retrieved content.
    """
    
    result = summarizer_agent(sample_content, source_label="RAG Research Paper")
    
    print("="*50)
    print(f"SOURCE: {result['source']}")
    print("="*50)
    print(result['summary'])