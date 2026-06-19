from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from orchestrator import orchestrator
import shutil
import os

app = FastAPI(
    title="MULTI-AGENT RESEARCH ASSISTANT",
    description="An AI system that researches any topic using multiple specialized agents.",
    version="1.0.0"
)

# valid request
class ResearchRequest(BaseModel):
    topic: str
    final_report: str
    used_pdf: bool

# valid response
class ResearchResponse(BaseModel):
    topic: str
    final_report: str

@app.get("/")
def health_check():
    """
    Simple endpoint to check if the API is running.
    """
    return {"status": "running", "message": "Multi-Agent Research Assistant API is live"}

@app.post("/research", response_model=ResearchResponse)
def research(topic: str = Form(...), pdf: UploadFile = File(None)):
    """
    Main endpoint - runs the full multi-agent pipeline on a given topic.
    """
    if not topic or len(topic.strip()) == 0:
        raise HTTPException(status_code=400, detail="Topic Cannot Be Empty!")
    
    pdf_path = None
    try:
        if pdf:
            os.makedirs("temp_uploads", exist_ok=True)
            pdf_path = f"temp_uploads/{pdf.filename}"
            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(pdf.file, buffer)
        
        result = orchestrator(topic, pdf_path=pdf_path)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
    
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            