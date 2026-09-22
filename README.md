# LexLens — AI Legal Document Instrument

A GenAI-powered assistant that makes legal documents easier to read, compare,
and act on — built on the Groq API (Llama models) for fast inference.

**This is not legal advice.** LexLens explains and organizes legal text; it
does not represent anyone or replace a licensed attorney. Every response
carries that reminder, and the assistant is explicitly instructed never to
tell a user what they should do — only what the document says.

## What it does

| Mode | What it produces |
|---|---|
| **Plain-language rewrite** | Section-by-section rewrite of the document at an 8th-grade reading level, streamed live |
| **Clause & risk analysis** | Structured breakdown: document type, parties, obligations/rights/risks/deadlines per clause, severity tags, red flags |
| **Compare two documents** | Side-by-side diff of two contracts/policies — what's only in A, only in B, and why each difference matters |
| **Ask questions** | Grounded Q&A chat that only answers from the uploaded document, and says so when the document is silent |
| **Checklist & next steps** | Action items, key dates, documents to gather, and pointed questions to bring to a real lawyer, plus a glossary of terms used |

Supports `.pdf`, `.docx`, `.txt`, or pasted text as input.

## Architecture

- **Backend** (`/backend`): FastAPI + the `groq` Python SDK. Stateless
  endpoints, one per mode, using `openai/gpt-oss-120b` by default —
  streaming for the rewrite/chat modes, `response_format: json_object` for
  the structured modes (analysis, compare, checklist). Documents are held in
  an in-memory dict, scoped to the running process (swap for a real DB plus
  auth before putting this in front of real users).
- **Frontend** (`/frontend/index.html`): a single self-contained HTML file —
  no build step. Talks to the backend over `fetch`, including streamed
  responses for the rewrite and chat modes.

## Run it

### Deploy to Vercel

The repository includes a Vercel Python function at `api/index.py`. In the
Vercel project settings, add these Environment Variables for Production:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Redeploy after adding the variables. The deployed frontend automatically uses
the same-origin `/api` routes; the API status indicator should then show that
Groq is configured.

### 1. Backend

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows PowerShell
pip install -r requirements.txt
copy backend\.env.example backend\.env
# edit .env and paste your key from https://console.groq.com/keys
python -m uvicorn backend.main:app --reload --port 8000
```

### 2. Frontend

Just open `frontend/index.html` in a browser (double-click it, or serve it
with any static server). The top-right field in the header lets you point it
at a different API base URL if you deploy the backend elsewhere (e.g. a
Render/Fly/Railway URL). When hosted over HTTP(S), it defaults to the current
site origin; when opened directly as a file, it defaults to
`http://localhost:8000`.

## Notes on the Groq integration

- Model is set via `GROQ_MODEL` in `.env` — defaults to
  `openai/gpt-oss-120b`. Choose another model returned by Groq's models
  endpoint if you need lower latency.
- Streaming endpoints (`/api/simplify`, `/api/chat`) use Groq's
  `stream=True` and forward raw text chunks to the browser as they arrive.
- Structured endpoints (`/api/analyze`, `/api/compare`, `/api/checklist`)
  use Groq's JSON mode (`response_format={"type": "json_object"}`) with a
  strict schema in the system prompt, then parse and validate the result
  server-side before returning it.

## Extending it

- Swap the in-memory `DOCS` dict for Postgres/SQLite plus per-user auth for
  multi-user deployments.
- Add OCR (e.g. via a vision-capable Groq model or a dedicated OCR step) for
  scanned PDFs with no extractable text layer.
- Chunk + embed long documents for retrieval instead of stuffing the whole
  text into context, once documents exceed the model's practical window.
- Add jurisdiction-aware prompting (ask the user their state/country) so
  red-flag detection can reference locally relevant norms — while keeping
  the "not legal advice" boundary intact.
