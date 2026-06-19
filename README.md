# 🔍 Multi-Agent Research Assistant

An AI-powered research system that automates the entire research workflow — searching the web, reading documents, fact-checking claims, and synthesizing structured reports — using a coordinated pipeline of specialized AI agents.

**🚀 [Try the live demo](https://huggingface.co/spaces/Swetha110607/multi-agent-research-assistant)**

---

## What it does

Researching a topic manually means opening multiple browser tabs, reading through articles, mentally cross-checking facts, and compiling notes — often taking 30-60 minutes per topic. This system automates that entire workflow.

Give it a topic (and optionally a PDF), and it will:
1. Search the web for relevant, current information
2. Optionally read and extract insights from an uploaded document
3. Summarize findings into clear key points
4. Fact-check claims against the original sources
5. Synthesize everything into a clean, structured report with citations

---

## Architecture

```
User Input (Topic + optional PDF)
            ↓
     Orchestrator
            ↓
  ┌─────────────────────────────┐
  │  Web Search Agent           │  → searches via Tavily, summarizes via Gemini
  │  Paper Reader Agent (RAG)   │  → reads PDFs, stores in ChromaDB, answers questions
  │  Summarizer Agent           │  → condenses findings into key points
  │  Fact Checker Agent         │  → verifies claims against source context
  └─────────────────────────────┘
            ↓
     Synthesis Agent
            ↓
  Final Structured Report (Overview, Key Findings, Confidence Notes, Sources)
```

Each agent has a single, well-defined responsibility. The Orchestrator sequences calls between them, passing the output of one agent as input to the next — this design improves reliability and transparency compared to a single large prompt, since each step can be inspected and verified independently.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini API |
| Agent Framework | LangChain |
| Web Search | Tavily API |
| Vector Store (RAG) | ChromaDB |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Deployment | Hugging Face Spaces (Docker) |

---

## Project Structure

```
multi-agent-research/
├── app.py                  # Local dev frontend (calls FastAPI backend via HTTP)
├── app_deploy.py            # Deployment frontend (calls orchestrator directly)
├── src/
│   ├── main.py               # FastAPI backend with /research endpoint
│   ├── orchestrator.py       # Pipeline sequencing + synthesis agent
│   ├── web_search_agent.py   # Web search + summarization
│   ├── paper_reader_agent.py # RAG pipeline for PDF documents
│   ├── summarizer_agent.py   # Condenses raw content into key points
│   └── fact_checker_agent.py # Verifies claims against context
├── data/                    # Sample PDFs for testing
└── requirements.txt
```

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/swetha110607/multi-agent-research-assistant.git
cd multi-agent-research-assistant
```

**2. Set up a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API keys**

Create a `.env` file in the root:
```
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

**5. Run the backend** (Terminal 1)
```bash
cd src
uvicorn main:app --reload
```

**6. Run the frontend** (Terminal 2)
```bash
streamlit run app.py
```

---

## Design Decisions Worth Noting

**Why a fixed pipeline instead of an LLM-driven orchestrator?** A fixed sequence (search → summarize → fact-check → synthesize) is more predictable, easier to debug, and easier to explain — appropriate for a first version. An LLM-driven router that dynamically decides which agents to call is a natural next step.

**Why two frontend files?** `app.py` talks to the FastAPI backend over HTTP, mirroring how production systems separate frontend and backend services. `app_deploy.py` calls the orchestrator directly as a Python function, since Hugging Face Spaces' free tier is optimized for single-process deployments.

---

## Known Limitations

- Each request takes 30-90 seconds due to sequential LLM calls across multiple agents
- Free-tier Gemini API rate limits can be hit during heavy testing
- Progress indicators in the UI are simulated for UX smoothness, not true real-time pipeline tracking (a future improvement could use streaming responses)

---

## Future Improvements

- LLM-driven dynamic agent routing instead of a fixed pipeline
- Real-time progress streaming from backend to frontend
- Support for multiple PDFs per research session
- Caching previously researched topics to reduce redundant API calls
