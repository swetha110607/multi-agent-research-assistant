import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# search tool
search_tool = TavilySearchResults(
    max_results = 5,
    tavily_api_key = os.getenv("TAVILY_API_KEY")
)

# inititialise model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

def web_search_agent(topic : str) -> dict:
    """
    Searches the web for a topic and returns summarized findings with sources.
    """
    print(f"Searching for : {topic}")

    # Raw results 
    raw_results = search_tool.invoke(topic)

    # readable text format
    formatted_results = ""
    sources = []
    for i, result in enumerate(raw_results, 1):
        formatted_results += f"\nSource {i}: {result.get('url', 'N/A')}\n"
        formatted_results += f"Content: {result.get('content', 'N/A')}\n"
        sources.append(result.get('url', 'N/A'))

    # summarise using gemini
    messages = [
        SystemMessage(content="""You are a research assistant. You will be given raw web search results.
        Summarize the key findings in clear bullet points. Be factual and concise.
        Do not add information that isn't in the search results."""),
        HumanMessage(content=f"Topic: {topic}\n\nSearch Results:\n{formatted_results}")
    ]

    response = llm.invoke(messages)

    return {
        "topic" : topic,
        "summary" : response.content,
        "sources" : sources
    }

# Test the agent
if __name__ == "__main__":
    result = web_search_agent("latest developments in retrieval augmented generation")
    
    print("\n" + "="*50)
    print(f"TOPIC: {result['topic']}")
    print("="*50)
    print("\nSUMMARY:")
    print(result['summary'])
    print("\nSOURCES:")
    for source in result['sources']:
        print(f"- {source}")