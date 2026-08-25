# api.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import ollama
import time
import psutil
from google import genai
from google.genai import types
import pdfplumber
import io
import os

# ─── LOAD API KEY ───
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY"):
                    GEMINI_API_KEY = line.strip().split("=", 1)[1]
    except:
        pass

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-flash-latest"

app = FastAPI()

# ─── FINANCE SYSTEM PROMPT ───
SYSTEM_PROMPT = """You are a financial assistant helping 
professionals in the finance and banking industry. 
Answer clearly and accurately. Only provide finance 
relevant information. If you are unsure, say so clearly."""

# ─── MODEL LISTS ───
LOCAL_MODELS = ["mistral", "llama3.2", "phi3:mini"]
CLOUD_MODEL = "gemini"
ALL_MODELS = LOCAL_MODELS + [CLOUD_MODEL]

# ─── REQUEST MODEL ───
class Question(BaseModel):
    text: str
    model: str

# ─── HOME ENDPOINT ───
@app.get("/")
def home():
    return {
        "message": "FinanceAI is running",
        "models_available": ALL_MODELS
    }

# ─── MAIN ASK ENDPOINT ───
@app.post("/ask")
def ask(q: Question):

    if q.model not in ALL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {ALL_MODELS}"
        )

    ram_before = psutil.virtual_memory().used
    start = time.time()

    # ─── LOCAL MODELS via Ollama ───
    if q.model in LOCAL_MODELS:
        response = ollama.chat(
            model=q.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q.text}
            ]
        )
        answer = response["message"]["content"]

    # ─── CLOUD MODEL via Gemini with retry ───
    else:
        answer = None
        for attempt in range(3):
            try:
                chat = gemini_client.chats.create(
                    model=GEMINI_MODEL,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT
                    )
                )
                response = chat.send_message(q.text)
                answer = response.text
                break
            except Exception as retry_error:
                if attempt < 2:
                    time.sleep(5)
                else:
                    answer = "Gemini is temporarily unavailable. Please try again in a moment."

    duration = round(time.time() - start, 2)
    ram_used = round(
        (psutil.virtual_memory().used - ram_before) / (1024 * 1024), 2
    )

    return {
        "answer": answer,
        "model": q.model,
        "seconds": duration,
        "ram_mb": ram_used,
        "is_local": q.model in LOCAL_MODELS
    }

# ─── DOCUMENT UPLOAD ENDPOINT ───
@app.post("/ask-document")
async def ask_document(
    file: UploadFile = File(...),
    question: str = Form(...),
    model: str = Form(...)
):
    if model not in ALL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {ALL_MODELS}"
        )

    contents = await file.read()
    document_text = ""
    page_count = 0

    try:
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    document_text += text + "\n"
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read PDF: {e}"
        )

    if not document_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this PDF."
        )

    prompt = f"""The user has uploaded a financial document. 
Using only the information in this document, answer the question clearly and accurately.
If the answer is not in the document, say so clearly.

DOCUMENT CONTENT:
{document_text[:8000]}

QUESTION:
{question}"""

    ram_before = psutil.virtual_memory().used
    start = time.time()

    if model in LOCAL_MODELS:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response["message"]["content"]

    else:
        answer = None
        for attempt in range(3):
            try:
                chat = gemini_client.chats.create(
                    model=GEMINI_MODEL,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT
                    )
                )
                response = chat.send_message(prompt)
                answer = response.text
                break
            except Exception as retry_error:
                if attempt < 2:
                    time.sleep(5)
                else:
                    answer = "Gemini is temporarily unavailable. Please try again in a moment."

    duration = round(time.time() - start, 2)
    ram_used = round(
        (psutil.virtual_memory().used - ram_before) / (1024 * 1024), 2
    )

    return {
        "answer": answer,
        "model": model,
        "seconds": duration,
        "ram_mb": ram_used,
        "is_local": model in LOCAL_MODELS,
        "pages_read": page_count
    }