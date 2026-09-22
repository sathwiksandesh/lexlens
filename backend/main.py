"""
LexLens — AI-powered legal document assistant.

Backend: FastAPI + Groq (Llama models).
Provides: document ingestion, plain-language simplification, clause/risk
analysis, document comparison, grounded Q&A, and action checklists.

This tool provides information and general assistance. It does not
provide legal advice and does not create an attorney-client relationship.
"""

import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

try:
    from groq import Groq
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set. Set it in backend/.env before making requests.")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Model choice: fast + capable, good for both structured JSON and long-form text.
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

app = FastAPI(title="LexLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(FRONTEND_INDEX)

# In-memory document store. Fine for a demo / single-session tool;
# swap for a real DB (with per-user auth) before any production use.
DOCS: dict[str, dict] = {}

MAX_CHARS = 60000  # keep prompts within a safe context budget

DISCLAIMER = (
    "LexLens explains and organizes legal text. It does not provide legal advice, "
    "does not represent you, and is not a substitute for a licensed attorney. "
    "For decisions with real consequences, confirm with a qualified professional "
    "in your jurisdiction."
)


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------

def extract_text(filename: str, raw: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    if name.endswith(".docx"):
        import docx

        d = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in d.paragraphs)
    # .txt / .md / fallback
    return raw.decode("utf-8", errors="ignore")


def clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_CHARS]


def require_client():
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on the server.",
        )


def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def get_doc(doc_id: str) -> dict:
    doc = DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found. Upload it again.")
    return doc


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = clean(extract_text(file.filename, raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc

    if not text or len(text) < 20:
        raise HTTPException(status_code=400, detail="No readable text found in that file.")

    doc_id = str(uuid.uuid4())
    DOCS[doc_id] = {"filename": file.filename, "text": text}
    return {"doc_id": doc_id, "filename": file.filename, "chars": len(text), "preview": text[:600]}


class TextIn(BaseModel):
    text: str
    filename: Optional[str] = "Pasted text"


@app.post("/api/upload-text")
async def upload_text(payload: TextIn):
    text = clean(payload.text)
    if not text or len(text) < 20:
        raise HTTPException(status_code=400, detail="Paste more text — that looks too short.")
    doc_id = str(uuid.uuid4())
    DOCS[doc_id] = {"filename": payload.filename or "Pasted text", "text": text}
    return {"doc_id": doc_id, "filename": DOCS[doc_id]["filename"], "chars": len(text), "preview": text[:600]}


# --------------------------------------------------------------------------
# Simplify (streaming plain-language rewrite)
# --------------------------------------------------------------------------

SIMPLIFY_SYSTEM = """You rewrite legal documents in plain language for a non-lawyer reader.
Go section by section, in the document's own order. For each section:
- Give it a short plain-language heading (what this part actually does).
- Explain it in 2-4 plain sentences, at an 8th-grade reading level.
- If it creates an obligation, a payment, a deadline, or a right the reader gives up, say so explicitly.
Use markdown with ## headings. Do not invent content that isn't in the document.
Do not give legal advice or tell the reader what to do — only explain what the text means.
End with a one-line reminder that this is not legal advice."""


class SimplifyIn(BaseModel):
    doc_id: str


@app.post("/api/simplify")
async def simplify(payload: SimplifyIn):
    require_client()
    doc = get_doc(payload.doc_id)

    def gen():
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SIMPLIFY_SYSTEM},
                {"role": "user", "content": f"Document: {doc['filename']}\n\n{doc['text']}"},
            ],
            stream=True,
            temperature=0.2,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(gen(), media_type="text/plain")


# --------------------------------------------------------------------------
# Analyze (structured clauses / risks / obligations)
# --------------------------------------------------------------------------

ANALYZE_SYSTEM = """You are a careful legal-document analyst producing a structured readout.
Analyze the document and return ONLY valid JSON (no markdown fences, no commentary), matching this shape exactly:

{
  "document_type": "short guess at what kind of document this is",
  "parties": ["party names or roles found in the text"],
  "summary": "2-3 sentence plain-language summary",
  "clauses": [
    {
      "title": "short label",
      "category": "obligation" | "right" | "risk" | "deadline" | "payment" | "termination" | "other",
      "severity": "low" | "medium" | "high",
      "excerpt": "short verbatim excerpt from the document, under 25 words",
      "plain_explanation": "1-2 plain sentences on what this means for the reader"
    }
  ],
  "key_dates": ["any deadlines, terms, renewal windows, or dated obligations found"],
  "red_flags": ["notably one-sided, unusual, or risky terms worth extra attention, plain language"]
}

Only use "high" severity for terms with real financial, legal, or rights-losing consequences.
Base everything strictly on the document text. If a field has nothing to report, use an empty array.
Return between 5 and 15 clauses, prioritizing the most consequential ones."""


class AnalyzeIn(BaseModel):
    doc_id: str


@app.post("/api/analyze")
async def analyze(payload: AnalyzeIn):
    require_client()
    doc = get_doc(payload.doc_id)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ANALYZE_SYSTEM},
            {"role": "user", "content": f"Document: {doc['filename']}\n\n{doc['text']}"},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {exc}") from exc

    data["disclaimer"] = DISCLAIMER
    return data


# --------------------------------------------------------------------------
# Compare two documents
# --------------------------------------------------------------------------

COMPARE_SYSTEM = """You compare two versions of a legal document (or two related documents,
e.g. two vendor contracts) for a non-lawyer reader. Return ONLY valid JSON, no markdown fences:

{
  "overview": "2-3 sentence plain-language summary of how the documents differ overall",
  "differences": [
    {
      "topic": "short label for what this difference is about",
      "doc_a": "what Document A says, plain language, or 'Not present'",
      "doc_b": "what Document B says, plain language, or 'Not present'",
      "significance": "low" | "medium" | "high",
      "why_it_matters": "1-2 plain sentences on the practical impact of this difference"
    }
  ],
  "only_in_a": ["notable terms present only in Document A"],
  "only_in_b": ["notable terms present only in Document B"],
  "recommendation_notes": ["neutral, non-advisory observations about what a reader may want to double check or ask about, not instructions to act"]
}

Focus on substantive differences (obligations, price, term length, liability, termination, rights given up),
not wording or formatting. Base everything strictly on the two texts provided."""


class CompareIn(BaseModel):
    doc_id_a: str
    doc_id_b: str


@app.post("/api/compare")
async def compare(payload: CompareIn):
    require_client()
    doc_a = get_doc(payload.doc_id_a)
    doc_b = get_doc(payload.doc_id_b)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": COMPARE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"=== Document A: {doc_a['filename']} ===\n{doc_a['text']}\n\n"
                    f"=== Document B: {doc_b['filename']} ===\n{doc_b['text']}"
                ),
            },
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {exc}") from exc

    data["disclaimer"] = DISCLAIMER
    return data


# --------------------------------------------------------------------------
# Grounded Q&A (streaming)
# --------------------------------------------------------------------------

CHAT_SYSTEM = """You answer questions about ONE specific legal document, for a non-lawyer.
Ground every answer strictly in the provided document text — quote or paraphrase the relevant
part when useful. If the document does not address the question, say so plainly rather than
guessing or filling in generic legal knowledge. Keep answers concise (under ~150 words) unless
the question needs more. Never tell the reader what they should legally do — describe what the
document says and let them decide, suggesting they consult a professional for anything consequential."""


class ChatIn(BaseModel):
    doc_id: str
    question: str
    history: list[dict] = []  # [{role: "user"|"assistant", content: str}, ...]


@app.post("/api/chat")
async def chat(payload: ChatIn):
    require_client()
    doc = get_doc(payload.doc_id)

    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": f"Document ({doc['filename']}):\n\n{doc['text']}"},
        {"role": "assistant", "content": "Understood. I'll answer questions grounded in this document only."},
    ]
    for turn in payload.history[-8:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": payload.question})

    def gen():
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            temperature=0.2,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(gen(), media_type="text/plain")


# --------------------------------------------------------------------------
# Checklist / next steps / questions for a lawyer
# --------------------------------------------------------------------------

CHECKLIST_SYSTEM = """Based on this legal document, produce ONLY valid JSON (no markdown fences):

{
  "action_items": ["concrete things the reader may need to do or watch for, e.g. 'Sign and return by [date]'"],
  "key_dates": ["dated or time-bound obligations found in the document"],
  "documents_to_gather": ["items the reader may need to prepare, e.g. ID, proof of address, prior agreements"],
  "questions_for_a_lawyer": ["specific, pointed questions the reader could bring to a licensed attorney about this document"],
  "glossary": [{"term": "legal term used in the document", "definition": "one-sentence plain definition"}]
}

Keep every item specific to this document, not generic advice. Include 3-8 items per list where applicable.
This is preparation material, not legal advice or instructions to act."""


class ChecklistIn(BaseModel):
    doc_id: str


@app.post("/api/checklist")
async def checklist(payload: ChecklistIn):
    require_client()
    doc = get_doc(payload.doc_id)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CHECKLIST_SYSTEM},
            {"role": "user", "content": f"Document: {doc['filename']}\n\n{doc['text']}"},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {exc}") from exc

    data["disclaimer"] = DISCLAIMER
    return data


@app.get("/api/health")
async def health():
    return {"status": "ok", "groq_configured": client is not None, "model": MODEL}
