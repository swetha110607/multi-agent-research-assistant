import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1  # very low - we want consistent, factual judgments, not creativity
)


def fact_checker_agent(claim: str, supporting_context: str) -> dict:
    """
    Checks a claim against provided context and returns a confidence assessment.
    """
    messages = [
        SystemMessage(content="""You are a rigorous fact checker. Given a claim and supporting context:
        
        1. Determine if the context supports, contradicts, or is silent on the claim
        2. Assign a confidence level: HIGH, MEDIUM, LOW, or UNVERIFIABLE
        3. Explain your reasoning in 2-3 sentences
        4. If the context doesn't mention the claim at all, say UNVERIFIABLE - do not guess
        
        Respond in this exact format:
        CONFIDENCE: [level]
        REASONING: [your explanation]"""),
        HumanMessage(content=f"Claim to verify: {claim}\n\nSupporting context:\n{supporting_context}")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "claim": claim,
        "assessment": response.content
    }


# Test the agent
if __name__ == "__main__":
    claim = "RAG systems eliminate hallucination completely in language models"
    context = """
    Retrieval Augmented Generation (RAG) combines the parametric knowledge of large 
    language models with non-parametric knowledge retrieved from external sources. 
    This approach addresses key limitations of standalone LLMs, including hallucination, 
    outdated knowledge, and lack of source attribution.
    """
    
    result = fact_checker_agent(claim, context)
    
    print("="*50)
    print(f"CLAIM: {result['claim']}")
    print("="*50)
    print(result['assessment'])