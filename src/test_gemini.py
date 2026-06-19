import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load API key from .env
load_dotenv()

# Initialise model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

# 1 --> Basic Output
print("Test 1: Basic Output")
response = llm.invoke("Explain RAG in 3 lines")
print(response.content)
print()

# 2 --> With system prompt
print("Test 2: With system prompt")
messages = [
    SystemMessage(content="You are a research assistant. Always respond in bullet points with sources when possible."),
    HumanMessage(content="What is tranformer architecure?")
]
response = llm.invoke(messages)
print(response.content)
print()

# 3 --> Chained thinking
print("Test 3: Step by step reasoning")
messages = [
    SystemMessage(content="You are an expert researcher.Think step by step before answering"),
    HumanMessage(content="What are the pros and cons of using RAG vs fine-tuning an LLM?")
]
response = llm.invoke(messages)
print(response.content)