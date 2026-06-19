import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from paper_reader_agent import paper_reader_agent

from web_search_agent import web_search_agent
from summarizer_agent import summarizer_agent
from fact_checker_agent import fact_checker_agent

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5
)

def synthesis_agent(topic: str, search_summary: str, fact_check_results: list, sources: list) -> str:
    """
    Combines all agent outputs into one final structured report.
    """
    fact_check_text = "\n".join([f"- {fc}" for fc in fact_check_results])
    sources_text = "\n".join([f"- {s}" for s in sources])

    messages = [
        SystemMessage(content="""You are a research report writer. Combine the provided information 
        into a clean, well-structured report with these sections:
        
        1. Overview (2-3 sentences introducing the topic)
        2. Key Findings (bullet points from the research)
        3. Confidence Notes (mention anything flagged as uncertain by fact-checking)
        4. Sources
        
        Be factual and do not add information beyond what's provided."""),
        HumanMessage(content=f"""
        Topic: {topic}
        
        Research Summary:
        {search_summary}
        
        Fact-Check Results:
        {fact_check_text}
        
        Sources:
        {sources_text}
        """)
    ]
    response = llm.invoke(messages)
    return response.content

def orchestrator(topic: str, pdf_path: str = None) -> dict:
    """
    Runs the full pipeline: search -> summarize -> fast-check -> synthesize.
    """
    print(f"\n Starting research pipline for : {topic}\n")

    # 1: Web search
    print("Step 1/4: Searching the web....")
    search_result = web_search_agent(topic)

    # optional
    pdf_insights = None
    if pdf_path:
        print("Extra step: Reading uploaded PDF....")
        pdf_result = paper_reader_agent(pdf_path, question=f"What does this document say about {topic}?")
        pdf_insights = pdf_result["answer"]


    # 2: Summarize
    print("Step 2/4: Summarizing findings....")
    combined_content = search_result["summary"]
    if pdf_insights:
        combined_content += f"\n\nAdditonal insights from uploaded document:\n{pdf_insights}"
    summary_result = summarizer_agent(
        content=combined_content,
        source_label=topic
    )

    # 3: Fast-check key claims from summary
    print("Step 3/4: Fast-checking key claims....")
    fact_check_result = fact_checker_agent(
        claim=summary_result["summary"],
        supporting_context=combined_content
    )

    # 4: Synthesize final report
    print("Step 4/4: Writing final report....")
    final_report = synthesis_agent(
        topic=topic,
        search_summary=summary_result["summary"],
        fact_check_results=[fact_check_result["assessment"]],
        sources=search_result["sources"]
    )

    print("PIPELINE COMPLETED!!\n")

    return {
        "topic" : topic,
        "final_report" : final_report,
        "used_pdf" : pdf_path is not None
    }

# Test the full pipeline
if __name__ == "__main__":
    result = orchestrator("impact of large language models on software engineering jobs")

    print("="*60)
    print("FINAL RESEARCH REPORT")
    print("="*60)
    print(result["final_report"])