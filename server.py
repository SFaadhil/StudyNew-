from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, shutil, re, requests
from pypdf import PdfReader
from typing import List

#  App Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

#  PAGE INDEX
#  { filename: [ {page, text}, ... ] }

PAGE_INDEX: dict[str, list[dict]] = {}


def clean_pdf_text(text: str) -> str:
    """Fix PDF extraction artifacts — join broken lines, preserve paragraphs."""
    text = re.sub(r'\n{2,}', '<<PARA>>', text)
    text = re.sub(r'\n', ' ', text)
    text = text.replace('<<PARA>>', '\n\n')
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def index_pdf(file_path: str, filename: str) -> int:
    """Extract text page-by-page, clean it, and store in PAGE_INDEX."""
    reader = PdfReader(file_path)
    pages  = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw  = (page.extract_text() or "").strip()
        text = clean_pdf_text(raw)
        if text:
            pages.append({"page": page_num, "text": text})
    PAGE_INDEX[filename] = pages
    return len(pages)


def rebuild_index_on_startup():
    for filename in os.listdir(UPLOAD_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(UPLOAD_DIR, filename)
            index_pdf(path, filename)


#  RETRIEVAL — frequency-weighted scoring
def normalize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def retrieve_pages(query: str, top_k: int = 3) -> list[dict]:
    """
    Score pages by keyword overlap + frequency bonus.
    Pages where query words appear many times score higher.
    """
    query_words = set(normalize(query))
    scored = []

    for filename, pages in PAGE_INDEX.items():
        for entry in pages:
            page_words     = normalize(entry["text"])
            page_word_set  = set(page_words)
            unique_matches = len(query_words & page_word_set)
            if unique_matches == 0:
                continue
            freq_bonus = sum(page_words.count(w) for w in query_words)
            score      = (unique_matches * 3) + freq_bonus
            scored.append({
                "file":  filename,
                "page":  entry["page"],
                "text":  entry["text"],
                "score": score
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


#  OLLAMA CALL
def call_ollama(prompt: str) -> str:
    try:
        res = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        res.raise_for_status()
        return res.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama is not running. Open a terminal and run: ollama serve"
    except Exception as e:
        return f"⚠️ Ollama error: {str(e)}"


#  PROMPTS
STRICT_PROMPT = """You are a strict study assistant. Answer ONLY using the source material below.
You are absolutely NOT allowed to use any outside knowledge.

CRITICAL RULES:
1. ONLY use facts explicitly written in the source material.
2. Cite sources using SHORT inline markers like: [p.6] or [p.3] — just the page number, nothing else.
   Do NOT write out full "(Source: filename, Page X)" — only use [p.X] format.
3. For specific details — numbers, durations, names, timeframes:
   - Clearly stated in source → quote it exactly, add [p.X]
   - Only partially covered → state what IS there, then write:
     ⚠️ Uncertain: the source does not fully specify [the detail]
4. NEVER invent or estimate anything not in the source.
5. If the topic is completely absent: "This topic isn't covered in your uploaded material."
6. FORMAT:
   - Write 2-3 paragraphs explaining what the source says, with [p.X] markers after cited facts
   - Follow with bullet points listing specific facts, rules, numbers, conditions
   - Each bullet ends with [p.X]

SOURCE MATERIAL:
{context}

QUESTION: {query}

ANSWER:"""


DETAIL_PROMPT = """You are a precise study assistant giving a DEEP, structured explanation.
Use ONLY the source material. Never add outside knowledge.

RULES:
1. Only use facts from the source. Cite using short [p.X] markers after each fact — NOT full "(Source: filename, Page X)".
2. For numbers, durations, names, deadlines:
   - Clearly in source → quote exactly, add [p.X]
   - Partially covered → show what IS there + flag: ⚠️ Uncertain: source does not fully specify [detail]
3. Never invent specifics.

FORMAT your answer with these exact markdown sections:

## Summary
2-3 sentences on what the source says about this topic.

## Key Points
Numbered list of every relevant detail. Cite page for each.

## Important Rules & Conditions
Timeframes, exceptions, who is responsible, consequences — cite page for each.
Flag incomplete info with ⚠️ Uncertain.

## Exact Clauses Referenced
Copy the most relevant clause numbers and text verbatim from the source.

SOURCE MATERIAL:
{context}

QUESTION: {query}

DETAILED ANSWER:"""


KID_PROMPT = """You are a friendly tutor explaining something to a 12-year-old student.
Use ONLY the information in the source material below — no outside knowledge.

RULES:
1. Only use facts from the source material.
2. Use very simple language — no jargon or complex terms.
3. When you use a technical term, immediately explain it in plain words in brackets.
4. Use fun, relatable analogies to explain concepts (like comparing rules to school rules).
5. Break everything into short, easy sentences.
6. Use emojis occasionally to make it friendly 😊
7. If something isn't in the source, say: "Hmm, my notes don't cover that part!"
8. Cite the source at the end: (From: filename, Page X)

SOURCE MATERIAL:
{context}

QUESTION: {query}

SIMPLE EXPLANATION:"""


SOURCE_EXPLAIN_PROMPT = """You are a study assistant. A student wants to see the source material
AND understand it at the same time.

Your job: for EACH section of the source material provided, write a clear explanation
of what that section means, directly next to it.

RULES:
1. Only use facts from the source. Never add outside knowledge.
2. Go through the source section by section.
3. For each section: first show the clause/text, then explain what it means in plain language.
4. Use this format for each section:
   **📄 Source (Page X):** [the relevant clause text verbatim]
   **💡 What this means:** [plain language explanation]
5. Cite page numbers throughout.
6. Flag anything unclear with ⚠️ Uncertain.

SOURCE MATERIAL:
{context}

QUESTION: {query}

SIDE-BY-SIDE ANSWER:"""

EXPAND_PROMPT = """You are a helpful and knowledgeable study assistant.
The student's uploaded material didn't fully cover their question.
Answer using your broader knowledge. Be clear, educational, and well-structured.
Always note when you are going beyond the uploaded material.

{partial_context}QUESTION: {query}

ANSWER:"""


#  AGENTS
def build_context(pages: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['file']} — Page {p['page']}]\n{p['text']}"
        for p in pages
    )

def make_sources(pages):
    return [{"file": p["file"], "page": p["page"]} for p in pages]

def make_raw(pages):
    return [{"file": p["file"], "page": p["page"], "text": p["text"]} for p in pages]


def strict_agent(query: str, pages: list[dict]) -> dict:
    source_reply = call_ollama(STRICT_PROMPT.format(context=build_context(pages), query=query))
    # Always generate an AI supplement so the user gets broader context automatically
    ai_reply = call_ollama(EXPAND_PROMPT.format(partial_context="", query=query))
    return {
        "reply":        source_reply,
        "ai_supplement": ai_reply,
        "sources":      make_sources(pages),
        "raw_pages":    make_raw(pages),
        "mode":         "strict"
    }


def detail_agent(query: str, pages: list[dict]) -> dict:
    reply = call_ollama(DETAIL_PROMPT.format(context=build_context(pages), query=query))
    return {"reply": reply, "sources": make_sources(pages), "raw_pages": make_raw(pages), "mode": "detail"}


def kid_agent(query: str, pages: list[dict]) -> dict:
    reply = call_ollama(KID_PROMPT.format(context=build_context(pages), query=query))
    return {"reply": reply, "sources": make_sources(pages), "raw_pages": make_raw(pages), "mode": "kid"}


def source_explain_agent(query: str, pages: list[dict]) -> dict:
    reply = call_ollama(SOURCE_EXPLAIN_PROMPT.format(context=build_context(pages), query=query))
    return {"reply": reply, "sources": make_sources(pages), "raw_pages": make_raw(pages), "mode": "source_explain"}


HYBRID_PROMPT = """You are a study assistant that blends two sources of knowledge in one response.

You have been given source material from the student's uploaded documents AND your own general knowledge.

STRUCTURE your response in exactly these two sections:

## 📄 From Your Material
Answer strictly from the source material below. Use [p.X] markers after each cited fact.
If the source only partially covers the topic, state what IS there and flag:
⚠️ Uncertain: the source does not fully specify [detail]

## 🌐 Broader Context (AI Knowledge)
Now expand with your own general knowledge on this topic.
Explain related concepts, background, real-world context, or anything that helps the student understand more deeply.
Clearly distinguish this from the source material — this is general knowledge, not from their documents.
Keep it focused and educational.

SOURCE MATERIAL:
{context}

QUESTION: {query}

BLENDED ANSWER:"""


def hybrid_agent(query: str, pages: list[dict]) -> dict:
    reply = call_ollama(HYBRID_PROMPT.format(context=build_context(pages), query=query))
    return {"reply": reply, "sources": make_sources(pages), "raw_pages": make_raw(pages), "mode": "hybrid"}



    partial = ""
    if pages:
        partial = "Partial context from uploaded material:\n" + build_context(pages) + "\n\n"
    reply = call_ollama(EXPAND_PROMPT.format(partial_context=partial, query=query))
    return {"reply": reply, "sources": make_sources(pages), "raw_pages": make_raw(pages), "mode": "expanded"}


#  REQUEST MODELS
class ChatPayload(BaseModel):
    message: str
    mode:    str = "strict"   # strict | expand | detail | kid | source_explain | hybrid


class RetrievePayload(BaseModel):
    query: str


#  ROUTES
@app.get("/")
def root():
    return {"status": "StudyNew backend running", "model": OLLAMA_MODEL,
            "indexed_files": list(PAGE_INDEX.keys())}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    if file.filename.lower().endswith(".pdf"):
        pages_indexed = index_pdf(file_path, file.filename)
        return {"filename": file.filename, "pages_indexed": pages_indexed, "status": "indexed"}
    return {"filename": file.filename, "status": "uploaded (non-PDF, not indexed)"}


@app.get("/files")
def list_files():
    result = []
    for filename in os.listdir(UPLOAD_DIR):
        pages = len(PAGE_INDEX.get(filename, []))
        result.append({"name": filename, "pages": pages})
    return result


@app.post("/chat")
def chat(payload: ChatPayload):
    query = payload.message.strip()
    if not query:
        return {"reply": "Please enter a question.", "sources": [], "mode": "error"}

    top_k = 4 if payload.mode in ("detail", "source_explain") else 3
    pages = retrieve_pages(query, top_k=top_k)

    if payload.mode == "expand":
        return expand_agent(query, pages)

    if not pages:
        # Nothing in source — auto-generate a hybrid answer and flag it clearly
        ai_reply = call_ollama(EXPAND_PROMPT.format(partial_context="", query=query))
        return {
            "reply":     ai_reply,
            "sources":   [],
            "raw_pages": [],
            "mode":      "not_in_source"
        }

    if payload.mode == "detail":
        return detail_agent(query, pages)
    elif payload.mode == "kid":
        return kid_agent(query, pages)
    elif payload.mode == "source_explain":
        return source_explain_agent(query, pages)
    elif payload.mode == "hybrid":
        return hybrid_agent(query, pages)
    else:
        return strict_agent(query, pages)


@app.post("/retrieve")
def retrieve(payload: RetrievePayload):
    pages = retrieve_pages(payload.query)
    return {"query": payload.query, "results": [
        {"file": p["file"], "page": p["page"], "score": p["score"], "preview": p["text"][:200]}
        for p in pages
    ]}


@app.get("/index")
def get_index():
    return {
        filename: [{"page": p["page"], "preview": p["text"][:100]} for p in pages]
        for filename, pages in PAGE_INDEX.items()
    }


@app.delete("/files/{filename}")
def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    if filename in PAGE_INDEX:
        del PAGE_INDEX[filename]
    return {"deleted": filename}


#  STARTUP
@app.on_event("startup")
def startup_event():
    rebuild_index_on_startup()
    print(f"✅ Page Index ready — {len(PAGE_INDEX)} file(s) indexed")
    for fname, pages in PAGE_INDEX.items():
        print(f"   • {fname}: {len(pages)} pages")
    print(f"🤖 Model: {OLLAMA_MODEL}  |  🌐 {OLLAMA_URL}")
