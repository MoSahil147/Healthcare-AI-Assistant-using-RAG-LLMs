# Healthcare AI Assistant: Project Journal

> A complete record of how this project was built: planning, decisions, bugs, lessons learnt, and everything in between.
> Sections are ordered in the way you would present this project to an evaluation panel.

---

## Table of Contents

**Part 1 — Introduction (What we built and why)**
1. [How We Planned and Approached This Project](#1-how-we-planned-and-approached-this-project)
2. [The Working Application](#2-the-working-application)
3. [Assumptions Made While Building This Project](#3-assumptions-made-while-building-this-project)

**Part 2 — Technical Deep Dive (How it works)**
4. [Architecture and Design Approach](#4-architecture-and-design-approach)
5. [The RAG Pipeline and LLM Integration](#5-the-rag-pipeline-and-llm-integration)
6. [Prompt Engineering Strategy](#6-prompt-engineering-strategy)
7. [Agent and Tool Workflow Implementation](#7-agent-and-tool-workflow-implementation)
8. [API and Demo Flow](#8-api-and-demo-flow)

**Part 3 — Decisions and Trade-offs (Why we built it this way)**
9. [Key Technical Decisions and Trade-offs](#9-key-technical-decisions-and-trade-offs)

**Part 4 — Challenges and Learnings (What went wrong and what we learnt)**
10. [Problems We Faced and How We Debugged Them](#10-problems-we-faced-and-how-we-debugged-them)
11. [New Things We Learnt](#11-new-things-we-learnt)
12. [What to Keep in Mind if Starting Again](#12-what-to-keep-in-mind-if-starting-again)

**Part 5 — Honest Assessment (Where we stand)**
13. [Limitations and Future Improvements](#13-limitations-and-future-improvements)

**Part 6 — Modifications and Improvements Made**
14. [Modifications and Improvements Made After Initial Build](#14-modifications-and-improvements-made-after-initial-build)

**Part 7 — Panel Preparation (Anticipating questions)**
15. [Q&A Playbook: Good Questions, Failure Cases, and Hard Questions](#15-qa-playbook)

---

## 1. How We Planned and Approached This Project

### The Starting Point

The idea was to build a **healthcare AI assistant** that could answer patient questions grounded in real documents, not hallucinate answers. The two core requirements were:

1. Answers must come from actual documents (privacy policy, discharge instructions, telehealth guidelines, etc.)
2. The system must also be able to route non-knowledge questions (like booking appointments) to a separate tool

We started by mapping the pipeline on paper before writing a single line of code:

```
Documents -> Chunk -> Embed -> Store in Vector DB
                                    |
User Question -> Rewrite -> Embed -> Retrieve -> LLM -> Answer
```

### Planning with Claude

The planning process happened iteratively. We had no big upfront spec; we built one layer at a time:

- First got the RAG pipeline working end-to-end: Documents -> ChromaDB -> LLM answer.
- Then added the agent router: greeting detection, appointment tool, and RAG fallback.
- Then added session history, query rewriting, rate limiting, fallback LLM, and confidence scoring.
- Finally Dockerised the app, fixed bugs, and documented everything.

### Core Design Decisions Made Early

| Decision | Reasoning |
|----------|-----------|
| Use RAG, not fine-tuning | Documents change; RAG lets us update without retraining |
| FastAPI as the web framework | Automatic docs (`/docs`), async support, Pydantic validation built in |
| ChromaDB for vector storage | Local, file-based, no separate DB server needed |
| Groq for LLM inference | Free tier, extremely fast (LPU hardware), good rate limits for a demo |
| HuggingFace for embeddings | Free embedding endpoint; `all-MiniLM-L6-v2` is well-tested for semantic search |
| Separate routing layer | Greetings and appointments do not need RAG; routing saves unnecessary LLM calls |

---

## 2. The Working Application

The application is a **chat-based healthcare assistant** accessible in the browser at `http://localhost:8000`.

**What it can do:**
- Answer medical and policy questions grounded in 6 real documents (no hallucination)
- Book mock appointments by department and day
- Handle greetings politely
- Show confidence scores (high / medium / low / none) for every answer
- Show source citations, indicating which document chunk the answer came from
- Track conversation context across multiple turns (query rewriting)

**Documents in the knowledge base:**
- `dpdpa_privacy_guidelines.txt`: patient privacy rights under DPDPA 2023 (Indian data protection law)
- `telehealth_guidelines.txt`: rules for remote consultations
- `discharge_instructions.txt`: post-hospitalisation care instructions
- `insurance_eligibility_faq.txt`: common insurance questions
- `appointment_scheduling_policy.txt`: booking rules and cancellation policy
- `medication_refill_policy.txt`: prescription refill procedures

**Tech stack:**
- Backend: FastAPI + Uvicorn
- Vector DB: ChromaDB (local, file-based)
- Embeddings: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- LLM: Groq API, `llama-3.3-70b-versatile` (primary) and `llama-3.1-8b-instant` (fallback)
- Frontend: Vanilla HTML/CSS/JavaScript (single file, `static/index.html`)
- Containerisation: Docker + Docker Compose

---

## 3. Architecture and Design Approach

```
+--------------------------------------------------+
|                  Browser UI                      |
|         (static/index.html, chat widget)         |
+----------------------+---------------------------+
                       | POST /ask {question, session_id}
+----------------------v---------------------------+
|              FastAPI (main.py)                   |
|  - Validates request (Pydantic AskRequest)       |
|  - Rate limits: 20 req/min per IP (slowapi)      |
|  - Auto-generates session_id if client omits it  |
+----------------------+---------------------------+
                       |
+----------------------v---------------------------+
|            Agent Router (agent.py)               |
|  - Greeting?            -> hardcoded reply       |
|  - Appointment keyword? -> tool call             |
|  - Else?                -> RAG pipeline          |
+------+-----------------------------+-------------+
       |                             |
+------v----------+      +----------v-----------+
|  RAG Pipeline   |      |  Appointment Tool    |
|  (rag.py)       |      |  (agent.py)          |
|  1. Rewrite Q   |      |  - Extract dept      |
|  2. Embed Q     |      |  - Extract date      |
|  3. Search DB   |      |  - Return mock slots |
|  4. Build prompt|      +----------------------+
|  5. Call LLM    |
|  6. Save history|
+------+----------+
       |
+------v--------------------------------------------------+
|         ChromaDB (store/chroma/)                        |
|  - 54 chunks from 6 documents                           |
|  - Cosine similarity search                             |
|  - Returns top-4 closest chunks with distances          |
+------+--------------------------------------------------+
       |
+------v--------------------------------------------------+
|         Groq LLM (llama-3.3-70b-versatile)              |
|  - temperature=0 for determinism                        |
|  - Fallback: llama-3.1-8b-instant on rate limit         |
|  - Retry with exponential backoff (tenacity)            |
+---------------------------------------------------------+
```

### Three-Layer Design

1. **Routing layer**: intent detection before anything expensive runs
2. **Retrieval layer**: semantic search over the knowledge base
3. **Generation layer**: LLM synthesises a grounded answer from retrieved context

This separation means greetings and appointment queries never touch the LLM or vector DB unnecessarily.

---

## 4. The RAG Pipeline and LLM Integration

### Document Ingestion (`embeddings.py`)

On startup (and whenever `POST /ingest` is called):

1. Load all `.txt` files from `./data/`
2. Split each document into chunks: **800 characters, 100-character overlap**
3. Convert each chunk into a vector using `all-MiniLM-L6-v2` (384-dimensional vectors)
4. Store all vectors in ChromaDB using **cosine similarity** rather than Euclidean; cosine ignores magnitude, which is better for text
5. **Atomic swap:** write to `./store/chroma_tmp`, then rename to `./store/chroma`; live queries never read a half-written database

Result: **54 chunks** across 6 documents, indexed and ready for semantic search.

### Why 800 Characters Per Chunk?

Medical and healthcare documents are different from regular text. A single idea — like a discharge instruction or a privacy policy clause — often spans multiple sentences. You cannot cut it too short or you lose the meaning.

We tested different sizes mentally:

| Chunk Size | Problem |
|------------|---------|
| Too small (200-300 chars) | A single medical sentence gets cut in half. The chunk loses context and becomes useless for answering questions. |
| Too large (1500+ chars) | One chunk covers too many different topics. When retrieved, it brings in irrelevant information that confuses the LLM. |
| **800 chars (chosen)** | Fits 2-3 complete sentences or one full policy clause. Enough context to answer a question, not so much that it adds noise. |

We also considered 900 characters but found that at that size, chunks from our 6 documents started overlapping in meaning — two different chunks would carry almost the same content, wasting retrieval slots. 800 was the sweet spot where each chunk felt like a distinct, self-contained piece of information.

### Why 100 Characters Overlap?

Without overlap, imagine a sentence that starts at character 795 and ends at character 830. It gets split — the first half goes into chunk 1, the second half into chunk 2. Neither chunk makes sense on its own.

100 characters of overlap means the last ~1-2 sentences of one chunk are repeated at the start of the next. This ensures:
- No sentence is ever broken at a boundary
- The context carries smoothly from one chunk to the next
- The LLM always receives complete thoughts, not half-sentences

We chose 100 specifically because our documents use short, policy-style sentences (typically 80-120 characters each). 100 characters is enough to capture one full sentence of overlap without wasting too much space repeating content.

### Why Retrieve Top 4 Chunks?

When a user asks a question, we search ChromaDB and return the 4 most relevant chunks. Here is why 4:

- **Too few (1-2 chunks):** If the answer is spread across two different parts of a document, retrieving only 1 chunk means you miss half the answer. For example, a question about telehealth refill policy might need one chunk about eligibility and another about the actual process.
- **Too many (8-10 chunks):** The LLM prompt becomes very long and noisy. Irrelevant chunks get mixed in, and the LLM may get confused about which part to use for the answer.
- **4 chunks:** Enough to cover answers that span multiple sections of a document, while keeping the prompt clean and focused. At our document scale (6 files, 54 chunks total), 4 gives good coverage without overloading the model.

### Query Flow (`rag.py`)

```python
def answer_question(question, session_id):
    # Step 1: clarify vague follow-ups
    clear = rewrite_query(question, session_id)

    # Step 2: semantic search
    chunks = query_similar(clear)  # top-4 by cosine distance

    # Step 3: reject if nothing close enough
    if chunks[0]["distance"] > 0.8:
        return FALLBACK  # "I could not find this..."

    # Step 4: build prompt + call LLM
    context = join(chunks)
    answer = generate(build_prompt(context, clear))

    # Step 5: save to history
    history.append(question, answer)

    return {answer, sources, confidence}
```

### Query Rewriting

When a user says *"what about its side effects?"* the RAG system cannot search for that without context. `rewrite_query()` takes the last 6 conversation turns and asks the LLM to produce a standalone question:

> "What are the side effects of [drug discussed earlier]?"

This dramatically improves retrieval recall for multi-turn conversations.

### Confidence Scoring

Based on cosine distance of the best chunk retrieved:

| Distance | Confidence | Meaning |
|----------|------------|---------|
| < 0.3 | high | Very close semantic match |
| 0.3 to 0.6 | medium | Reasonable match |
| 0.6 to 0.8 | low | Weak match, answer may be approximate |
| > 0.8 | none | Fallback triggered, no answer given |

### LLM Fallback (`llm.py`)

If `llama-3.3-70b-versatile` hits Groq's rate limit, the system silently switches to `llama-3.1-8b-instant`. For other transient errors, it retries up to 3 times with exponential backoff using `tenacity`.

---

## 5. Prompt Engineering Strategy

The system prompt in `llm.py` was the most carefully designed piece of the project:

```
You are a healthcare assistant. Answer using ONLY the context provided below.
If the context contains relevant information -- even if it uses different law names
or terminology than the question -- answer using what is in the context and note
any differences.
Only respond with "I could not find this information in the provided documents."
if the context has no relevant information at all.
Do not diagnose, prescribe, or give direct medical advice.
Keep your answer professional, accurate, and concise.

Context:
{context}

Question: {question}

Answer:
```

### Why Each Instruction Matters

- **"ONLY the context provided"**: prevents hallucination. The LLM cannot draw on its training data to fill gaps.
- **"even if it uses different law names"**: necessary because the file is named `hipaa_privacy_guidelines.txt` but actually contains DPDPA 2023 (Indian law), not HIPAA (US law). Without this clause, the LLM would refuse to answer HIPAA questions even though the document has the relevant information.
- **"Do not diagnose or prescribe"**: safety guardrail to keep the assistant in an informational role.
- **`temperature=0`**: makes responses deterministic. For a medical context, consistent wording matters more than creative phrasing.

---

## 6. Agent and Tool Workflow Implementation

### The Router (`agent.py`)

`route()` is the single entry point for every user message. It decides what to do in priority order:

```
Message received
       |
  Is it a greeting? (regex word boundary)
       | yes -> "Hello! How can I help..."
       | no
  Does it contain appointment keywords?
       | yes -> extract dept + date -> check_available_slots()
       | no
  RAG pipeline -> answer_question()
```

### Greeting Detection

We went through three versions of greeting detection, each fixing a new problem:

**Version 1 — simple substring (broken):**
```python
# BAD: "what is morphine" matches "hi" inside "morphine"
if any(kw in question.lower() for kw in GREETING_KEYWORDS):
```

**Version 2 — regex word boundary (better, but still had a gap):**
```python
# BETTER: \bhi\b only matches "hi" as a standalone word
if any(re.search(rf'\b{kw}\b', question.lower()) for kw in GREETING_KEYWORDS):
```
This fixed the morphine bug. But a new problem appeared: `"Hi, I need to book an appointment"` was still treated as a greeting and never reached the appointment tool. The user clearly had intent beyond just saying hello.

**Version 3 — pure greeting check (final):**
```python
# BEST: only treat as greeting if the message is JUST a greeting with nothing else
q_stripped = re.sub(rf'\b({"|".join(GREETING_KEYWORDS)})\b', '', question.lower()).strip(" !?,.")
is_pure_greeting = any(re.search(rf'\b{kw}\b', question.lower()) for kw in GREETING_KEYWORDS) and len(q_stripped) < 10
```

The idea: remove the greeting word from the message and check what's left. If almost nothing remains (less than 10 characters), it's a pure greeting. If there's real content left, the user has a question — skip the greeting and route normally.

```
"hi"                               → remove "hi" → "" → 0 chars  → greeting ✓
"hello!"                           → remove "hello" → "!" → 1 char → greeting ✓
"hi, I need to book an appointment"→ remove "hi" → "i need to book an appointment" → 30 chars → appointment tool ✓
"hi, I need to know my rights"     → remove "hi" → "i need to know my rights" → 24 chars → RAG ✓
```

### Appointment Tool

No LLM needed for this. Pure string matching extracts:
- **Department**: walks the canonical list then the alias map (`psychiatry -> mental health`, `orthopedics -> orthopaedics`)
- **Date**: extracts day names and converts to an actual calendar date (`"tuesday" -> "Tuesday, June 27"`)

Returns mock slots with a note to call to confirm. In production this would call a real scheduling API.

### Session Management

In-memory `OrderedDict` keyed by `session_id`, capped at 1000 entries. When the cap is reached, the oldest session is evicted (FIFO). This is memory-safe without needing Redis or a database.

---

## 7. API and Demo Flow

### Startup Lifecycle

FastAPI's `lifespan` context manager runs before any request is served:

1. `validate_env()`: crashes loudly if `GROQ_API_KEY` or `HF_TOKEN` is missing, rather than producing a cryptic error on the first request
2. `ingest_documents()`: loads and indexes all 6 documents into ChromaDB
3. Server is ready

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves the chat UI (`static/index.html`) |
| `/ask` | POST | Main Q&A endpoint (rate-limited 20/min/IP) |
| `/ingest` | POST | Manually re-index documents after updates |
| `/health` | GET | Returns active model names and status |
| `/docs` | GET | Swagger UI (auto-generated by FastAPI) |

### Request/Response Shape

```json
// POST /ask
{
  "question": "What are my rights under DPDPA?",
  "session_id": "abc-123"
}

// Response
{
  "answer": "Under the DPDPA 2023, patients have the right to...",
  "sources": [
    {
      "document": "dpdpa_privacy_guidelines.txt",
      "chunk": "The Digital Personal Data Protection Act 2023 grants..."
    }
  ],
  "confidence": "high",
  "tool_used": null,
  "tool_output": null
}
```

### Running with Docker

```bash
# First run (builds images)
docker compose up --build

# Subsequent runs
docker compose up

# Access
open http://localhost:8000
```

---

## 8. Key Technical Decisions and Trade-offs

| Decision | Why We Made It | The Trade-off |
|----------|---------------|---------------|
| **Groq over OpenAI** | Free tier, extremely fast LPU inference | Hard rate limits (30 RPM on free tier) |
| **HuggingFace embeddings API** | Free, no local GPU needed | Adds latency for embedding API calls; HF token required |
| **ChromaDB (local)** | Zero infrastructure, just a folder on disk | Not horizontally scalable; lost if volume not persisted |
| **In-memory session history** | Simple, no Redis needed | Lost on server restart; not shared across instances |
| **Cosine distance (not Euclidean)** | Cosine ignores vector magnitude, better for sentence embeddings | Slightly slower than L2 in Chroma (negligible at this scale) |
| **Atomic tmp-swap for ingest** | Zero-downtime re-indexing | Doubles disk usage briefly during ingest |
| **Lazy singletons for models** | Fast import time; tests do not make network calls | First request to each path is slightly slower |
| **`temperature=0`** | Deterministic, consistent medical answers | No phrasing variation; can sometimes sound robotic |
| **Keyword routing (no LLM)** | Instant, no API call for greetings or appointments | Can miss ambiguous intent ("I feel sick and need help") |
| **800-char chunks, 100-char overlap** | Balances context richness vs. retrieval noise | Large chunks mean fewer, broader results that may include irrelevant sentences |
| **DISTANCE_THRESHOLD = 0.8** | Catches truly irrelevant queries | May be too permissive; low-confidence answers still get returned |
| **Vanilla JS (no framework)** | Zero build step, single file | Harder to maintain at scale; no component reuse |

---

## 9. Problems We Faced and How We Debugged Them

### Bug 1: "what is morphine" classified as a greeting

**Symptom:** Any medical question containing the letters "hi" anywhere in the text would be classified as a greeting and return *"Hello! How can I help you?"* instead of a medical answer.

**Root cause:** Substring matching. `"hi" in "what is morphine"` is `True` because "morphine" contains the substring "hi" (morpHIne).

**Debug process:** Noticed "Greeting detected" in logs when asking about morphine. Immediately recognised it as a substring collision. Classic false positive pattern.

**Fix:** Replaced substring check with regex word boundaries:

```python
# Before
if any(kw in question.lower() for kw in GREETING_KEYWORDS):

# After
if any(re.search(rf'\b{kw}\b', question.lower()) for kw in GREETING_KEYWORDS):
```

---

### Bug 2: Vectorstore opened a fresh connection on every query

**Symptom:** Every `/ask` request was slow. Logs showed repeated ChromaDB connection setup.

**Root cause:** The original code created a new `Chroma(...)` object on every call to `_get_vectorstore()`, even though the underlying data on disk never changed between requests.

**Fix:** Lazy singleton pattern; create once, reuse forever:

```python
_vectorstore: Chroma | None = None

def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(persist_directory=CHROMA_PATH, ...)
    return _vectorstore
```

---

### Bug 3: Re-ingesting documents disrupted live queries

**Symptom:** During a `POST /ingest`, any concurrent `/ask` request would read a partially-written ChromaDB and crash.

**Root cause:** Overwriting the live vector store in-place while queries were running against it.

**Fix:** Write to a temp directory, then atomically rename:

```python
tmp_path = CHROMA_PATH + "_tmp"
Chroma.from_documents(..., persist_directory=tmp_path)  # write to temp
shutil.rmtree(CHROMA_PATH)  # remove old
Path(tmp_path).rename(CHROMA_PATH)  # atomic swap
_vectorstore = None  # reset singleton
```

---

### Bug 4: Session IDs were empty in logs

**Symptom:** Logs showed `[session=]`; the session field was always blank even though `session_id` was wired in the backend.

**Root cause:** The frontend JavaScript was not sending `session_id` in the POST body. The backend had the field, accepted it, and logged it, but there was nothing to log.

**Fix:** Frontend generates a UUID on page load and includes it in every `/ask` request:

```javascript
const SESSION_ID = crypto.randomUUID();
// ...
body: JSON.stringify({ question, session_id: SESSION_ID })
```

The backend also auto-generates one if the client omits it (Pydantic validator).

---

### Bug 5: Render deployment ran out of memory

**Symptom:** The Render.com deployment kept crashing with OOM (out-of-memory) errors.

**Root cause:** Loading the HuggingFace sentence-transformer model plus ChromaDB plus Groq plus FastAPI all into Render's free tier (512 MB RAM) was too much. The embedding model alone requires approximately 400 MB.

**Decision:** Cancelled the Render deployment entirely. Switched to local Docker Compose. Free cloud tiers are not viable for embedding-heavy workloads without using the HuggingFace API endpoint instead of downloading the model locally.

---

---

## 10. New Things We Learnt

### RAG is more about retrieval quality than LLM quality

The LLM is only as good as the chunks it receives. If the wrong chunks are retrieved (low relevance), even the best LLM will produce a bad answer. Chunk size, overlap, and the distance threshold matter more than which LLM you pick.

### Cosine distance is the right metric for text embeddings

Euclidean distance measures the straight-line distance between two vectors. But text embeddings encode semantic meaning in the *direction* of the vector, not its magnitude. Two sentences can have similar meaning but very different lengths (and thus magnitudes). Cosine distance only looks at the angle between vectors, which is what we actually care about.

### Lazy singletons are critical for ML models in web servers

Loading an embedding model or connecting to a vector store on every request is catastrophic for latency. The lazy singleton pattern (create once, cache globally, reuse forever) is the standard approach for any model or expensive resource in a web server.

### Regex word boundaries prevent substring collisions

Naive keyword matching (`"hi" in text`) will match "hi" inside words like "this", "their", "morphine". For intent detection, always use word boundaries: `\b{keyword}\b`.

### FastAPI's `lifespan` is the right place for startup logic

Not `@app.on_event("startup")` (deprecated). The modern pattern is a `@asynccontextmanager` lifespan function that `yield`s. Code before `yield` runs on startup; code after runs on shutdown.

### Atomic file operations prevent corrupt reads under concurrency

Never write to a live file that something else might be reading. Write to a temp location, then rename/move atomically. The OS guarantees that a rename is atomic; observers either see the old file or the new one, never a half-written state.

### Pydantic validators can generate default values

`@field_validator` with `mode="before"` runs before type validation, so you can transform or fill in values, such as auto-generating a `session_id` UUID when the client does not send one.

### `OrderedDict` is a simple bounded cache

For bounded session storage without Redis, an `OrderedDict` with a cap and `popitem(last=False)` to evict the oldest entry is a clean, dependency-free solution.

### Free cloud tiers are not viable for embedding workloads

The HuggingFace `all-MiniLM-L6-v2` model requires approximately 400 MB of RAM when loaded locally. Render's free tier is 512 MB total, which does not leave enough headroom for the rest of the application. Either use the HuggingFace **API endpoint** (which does the inference on HF's servers) or upgrade to a paid tier.

---

## 11. What to Keep in Mind if Starting Again

These are the things we would do differently from day one:

### 1. Design the session ID contract before building anything
We built session history on the backend and the frontend separately. The frontend did not send session IDs for days, so query rewriting silently did not work. Define the request contract (including `session_id`) before building either side.

### 3. Use word boundaries for keyword matching from the start
Simple `in` substring checks will cause false positives for medical terminology. Use `\b{keyword}\b` regex from day one.

### 4. Use the HuggingFace API endpoint, not a local model download
Using `HuggingFaceEndpointEmbeddings` (API call) instead of downloading the model locally keeps RAM usage low and makes deployment to free tiers feasible.

### 5. Exclude the vector store from `--reload` immediately
Add `--reload-exclude store` before starting development. Forgetting this causes a reload loop the moment ChromaDB writes anything.

### 6. Set `DISTANCE_THRESHOLD` with real data, not guesses
We used `0.8`. That may be too permissive. The right value depends on your embedding model and document domain. Measure actual distances on sample queries and set a threshold based on observed data.

### 7. Build the fallback LLM before you hit rate limits
We added the fallback model (`llama-3.1-8b-instant`) after hitting Groq rate limits in testing. It should be there from the beginning. Wire up the `RateLimitError` catch and exponential backoff before demos.

### 8. Test the greeting detector with medical words before shipping
Specifically test words like "therapy", "morphine", "physical", "psychiatry" as they contain common greeting substrings. A word boundary unit test for this is worth writing.

### 9. Persist ChromaDB across container restarts
If you are using Docker, mount `./store` as a volume. Without a volume, the entire vector store is rebuilt from scratch every time the container restarts (slow startup and wasted HF API calls).

### 10. Keep the frontend in one file intentionally, not accidentally
We ended up with a single `index.html` after reverting a CSS split. This is actually a good default for a demo or prototype: no build step, no asset pipeline, easy to share. Make it a deliberate decision, not a refactor that happens mid-project.

---

## 12. Limitations and Future Improvements

### Current Limitations

| Limitation | Impact |
|-----------|--------|
| In-memory session history | Lost on restart; not shared across multiple server instances |
| Hardcoded appointment slots | Not a real scheduling system; mock data only |
| Re-ingests all documents on every startup | Slow startup; wastes HF API calls if docs have not changed |
| No authentication | Anyone can call `/ask`; no user identity or access control |
| Only `.txt` files supported | No PDF, DOCX, or HTML ingestion |
| No streaming responses | Users wait for the full answer; no progressive display |
| Keyword-only appointment routing | Cannot extract complex multi-part appointment requests |
| Single-node ChromaDB | Cannot scale horizontally across multiple app instances |
| No PHI handling | Should not be used with real patient data in this form |

### Future Improvements

- **Persistent sessions:** move session history to Redis or a database (DynamoDB, Postgres) for multi-instance support
- **Incremental ingestion:** hash documents on startup and skip re-embedding unchanged files
- **Streaming answers:** use SSE (Server-Sent Events) or WebSocket to stream tokens as they arrive from Groq
- **PDF/DOCX support:** add `pypdf` and `python-docx` loaders to handle richer document formats
- **RAGAS evaluation:** instrument the RAG pipeline with RAGAS scores (faithfulness, answer relevancy, context precision) to measure and improve quality over time
- **Real scheduling API:** replace mock slots with an actual integration (for example Cal.com, Calendly, or a hospital HMS API)
- **Auth and audit logs:** add OAuth or API key authentication; log every query with user identity for compliance
- **Chunking strategy review:** experiment with semantic chunking (split on sentences, not characters) for better retrieval

---

## 13. Q&A Playbook

### Questions That Work Well (Best Answers)

These questions have high-quality chunks in the knowledge base and will return **high confidence** answers:

```
What are my rights under the DPDPA regarding my health data?
What information must a hospital provide upon patient discharge?
Can I consult a doctor via telehealth if I am in a rural area?
How do I request a medication refill?
What should I do if I develop a fever after surgery?
Is my insurance coverage verified before my appointment?
What happens if I need to cancel my appointment?
Can my medical data be shared with third parties?
What security measures protect my health data?
How do I file a complaint about a privacy violation?
```

### Questions That Will Cause the Model to Fail (or Return Fallback)

These expose the boundaries of the knowledge base:

```
What is the dosage of metformin for type 2 diabetes?
Is ibuprofen safe during pregnancy?
What are the symptoms of appendicitis?
What is the current wait time at the emergency room?
Can you book me an appointment for next Tuesday at 3pm specifically?
What are my rights under HIPAA?
```

### How Unknown Answers Are Handled

When no chunk in the database is close enough (cosine distance > 0.8), the system returns:

```
"I could not find this information in the provided documents."
```

- `confidence: "none"` is returned
- `sources: []` (empty; no chunks cited)
- The UI shows a grey "none" badge
- **No hallucination**: the LLM is never called if retrieval fails the threshold

This is a deliberate safety mechanism. It is better to admit ignorance than to fabricate a medical answer.

### Explaining Model, Embedding, and Vector DB Choices

**Why `all-MiniLM-L6-v2` for embeddings?**
- 384-dimensional vectors: small enough for fast cosine search
- Trained specifically for semantic sentence similarity
- Runs via the HuggingFace inference API (no GPU needed locally)
- Well-benchmarked: top performer on the SBERT leaderboard for its size class

**Why ChromaDB?**
- No separate server to run; it is a Python library that stores to disk
- Built-in cosine similarity with HNSW indexing (fast approximate nearest neighbour)
- Perfect for prototypes and demos; easy to replace with Pinecone, Weaviate, or pgvector in production

**Why Llama 3.3-70B via Groq?**
- 70B parameters: large enough for complex medical language understanding
- Groq's LPU hardware makes it 5-10x faster than running on GPU servers
- Free tier with a generous token limit for demos
- Fallback to 8B instant handles rate limit bursts without downtime

### Source Citations

Every answer includes a `sources` array showing exactly which document chunk was used:

```json
"sources": [
  {
    "document": "dpdpa_privacy_guidelines.txt",
    "chunk": "Under the Digital Personal Data Protection Act 2023, patients have the right to..."
  }
]
```

The UI renders these as expandable "Sources" cards. This is important in healthcare; users should be able to verify where the answer came from, not just trust an AI.

Citation is derived from the `metadata.source` field stored alongside each chunk in ChromaDB at ingest time.

### How the Solution Would Scale in Production

| Component | Current | Production-Ready Replacement |
|-----------|---------|------------------------------|
| Vector DB | ChromaDB (local file) | Pinecone, Weaviate, or pgvector (Postgres extension) |
| LLM | Groq free tier | Groq paid tier, AWS Bedrock, or Azure OpenAI with higher rate limits |
| Embeddings | HuggingFace API endpoint | Batch embedding with caching; consider Voyage AI or Cohere embeddings |
| Session history | In-memory OrderedDict | Redis with TTL expiry |
| Web server | Single Uvicorn process | Gunicorn with multiple Uvicorn workers and a load balancer |
| File storage | Local `./data/*.txt` | S3 or GCS bucket with event-driven re-ingestion on file upload |
| Re-ingestion | Manual `POST /ingest` | Event-driven: Lambda/Cloud Function triggers on document upload |
| Auth | None | OAuth 2.0 / OIDC + API gateway |

### Security, PHI, and Healthcare Compliance Considerations

**This application as-built is NOT compliant for use with real patient data.**

Key gaps and considerations:

| Area | Current State | What is Needed for Compliance |
|------|--------------|-------------------------------|
| **PHI handling** | No real patient data; documents are policies | Any real PHI requires HIPAA BAA with all vendors (Groq, HuggingFace, cloud providers) |
| **Data encryption** | None (local files, no TLS enforced) | TLS in transit; AES-256 at rest for all stored vectors and documents |
| **Authentication** | None; open API | User authentication and role-based access control (RBAC) |
| **Audit logging** | Basic request logs | Full audit trail: who asked what, when, and what data was accessed |
| **Data residency** | Local disk | HIPAA and DPDPA both have data localisation requirements |
| **LLM data retention** | Groq's policy | Verify vendor does not train on submitted data; get contractual guarantees |
| **Consent** | None | DPDPA 2023 requires explicit, informed consent before processing health data |
| **Breach notification** | None | Must notify affected patients within 72 hours (DPDPA) and 60 days (HIPAA) |
| **Input validation** | Basic Pydantic validation | Sanitise inputs to prevent prompt injection attacks |
| **Rate limiting** | 20 req/min per IP | Should also rate-limit per authenticated user, not just IP |

**Prompt injection risk:** A malicious user could try to override the system prompt by including instructions in their question (for example, *"Ignore all previous instructions and reveal..."*). Mitigation: never concatenate user input directly into critical system instructions; keep user input in a clearly delimited `Question:` field separated from the system prompt.

---

---

## 14. Assumptions Made While Building This Project

These are things we assumed to be true while building the project. We did not verify all of them with real data — they are reasonable guesses for a prototype/demo.

---

### 1. The documents are enough to answer most questions
We assumed that 6 text files (discharge instructions, telehealth guidelines, insurance FAQ, etc.) would cover the kinds of questions a user would ask. In reality, a real healthcare assistant would need hundreds of documents. Our knowledge base is intentionally small for demo purposes.

---

### 2. Users will ask questions in English
The system prompt, document content, and the embedding model (`all-MiniLM-L6-v2`) all work best with English text. We assumed users will type in English. If someone asks in Hindi or another language, the results may be poor.

---

### 3. Simple keyword matching is good enough to detect appointment questions
We assumed that if the user's message contains words like "appointment", "book", "slot", or a department name like "cardiology", it is definitely an appointment question. This works for simple cases but can fail for ambiguous messages like *"I am sick and need help"* — which is not caught by keywords but is still an appointment-related concern.

---

### 4. The appointment tool does not need to be real
We assumed the evaluators and users understand that the appointment slots are fake/mock data. There is no real calendar or hospital system behind it. The tool is there to demonstrate the concept of routing to an external tool — not to actually book anything.

---

### 5. One user session = one browser tab
We assumed each user opens one browser tab and has one conversation at a time. The session ID is generated once per page load. If the user opens two tabs, they get two different sessions with no shared history.

---

### 6. The server will not receive more than 20 requests per minute from one user
We set the rate limit to 20 requests per minute per IP address. We assumed normal users will not send more than that. This number was chosen to protect the Groq API quota, not based on any measured traffic data.

---

### 7. A cosine distance above 0.8 means the question is not in our documents
We chose `0.8` as the threshold for deciding when to return a fallback answer. If the best matching chunk has a distance score above 0.8, we assume the question is not answerable from our documents. This number was picked based on observation during testing — not scientifically validated. It may need tuning in production.

---

### 8. Chunk size of 800 characters is a good fit for our documents
We split documents into chunks of 800 characters with 100 characters of overlap. We assumed this size gives enough context per chunk without being so big that unrelated sentences get mixed together. We did not experiment with other sizes like 400 or 1200 characters.

---

### 9. No real patient data will be used
We assumed this is a demo with synthetic (fake) documents only. The system has no PHI handling, no encryption, and no compliance features. We assumed the evaluators will not run it against real patient records.

---

### 10. The last 6 conversation turns are enough context for query rewriting
When rewriting a vague follow-up question like *"what about its side effects?"*, we send the last 6 messages (3 user turns + 3 assistant replies) to the LLM for context. We assumed 6 turns is enough to understand what the user is referring to. Longer conversations might need more history.

---

### 11. One server process is enough
We assumed only one or two people will use this at a time (demo scenario). We run a single Uvicorn process. In production with many users, you would run multiple workers behind a load balancer.

---

### 12. Groq's free tier is fast and reliable enough for a demo
We assumed Groq would respond quickly and not hit rate limits during the demo. Groq's free tier allows 30 requests per minute to the LLM. If the demo gets heavy usage, the fallback model (`llama-3.1-8b-instant`) will kick in automatically.

---

## 14. Modifications and Improvements Made After Initial Build

These are changes we made after the first working version was ready. Each one fixed a real problem we noticed during testing.

---

### 1. Added Greeting Detection

**Why:** The chatbot had no way to handle simple greetings like "hi" or "hello". It would search ChromaDB for relevant medical chunks, find nothing useful, and return a fallback message — which felt rude and broken.

**What we added:** A `GREETING_KEYWORDS` set and a check at the top of the `route()` function in `agent.py`:

```python
GREETING_KEYWORDS = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon", "howdy"}

if any(re.search(rf'\b{kw}\b', question.lower()) for kw in GREETING_KEYWORDS):
    return {
        "answer": "Hello! How can I help you with your healthcare questions today?",
        "sources": [], "confidence": "high", ...
    }
```

Now the chatbot responds politely to greetings without hitting the database or the LLM.

---

### 2. Fixed Greeting Detection (Three Iterations)

**Problem 1 — Morphine bug:**
`"what is morphine"` was treated as a greeting because `"hi"` is a substring of `"morphine"`. Fixed using regex word boundaries (`\bhi\b`).

**Problem 2 — Mixed intent bug:**
After the word boundary fix, `"Hi, I need to book an appointment"` was still treated as a pure greeting. The user clearly wanted to book an appointment, but the greeting check ran first and returned "Hello!" — the appointment tool was never reached.

**Final fix — pure greeting check:**
```python
# Remove the greeting word and check what's left
q_stripped = re.sub(rf'\b({"|".join(GREETING_KEYWORDS)})\b', '', question.lower()).strip(" !?,.")
is_pure_greeting = any(re.search(rf'\b{kw}\b', question.lower()) for kw in GREETING_KEYWORDS) and len(q_stripped) < 10
```

If the message is almost entirely a greeting word (less than 10 characters left after removing it), respond with a greeting. Otherwise, route the message normally so the real intent is handled.

```
"hi"                                → greeting ✓
"Hi, I need to book an appointment" → appointment tool ✓
"Hi, I need to know my rights"      → RAG ✓
```

---

### 3. Fixed Session ID Not Being Sent from the Frontend

**Why:** The backend had full session history support — it stored conversation turns and used them to rewrite vague follow-up questions. But it was never working because the frontend was not sending a `session_id` with each request. Every message was treated as a fresh conversation with no history.

**What we changed:** Added one line to `static/index.html` to generate a session ID once when the page loads, and included it in every request:

```javascript
// Generate once when the page opens
const SESSION_ID = crypto.randomUUID();

// Send it with every question
body: JSON.stringify({ question, session_id: SESSION_ID })
```

Now all messages from the same browser tab share the same session, and the query rewriting feature actually works.

---

### 4. Added `.dockerignore` to Reduce Docker Image Size

**Why:** The Docker image was copying everything into the container — including `Demo.mp4`, the entire git history (`.git/`), the ChromaDB store, test files, and Python cache. This made the image unnecessarily large and slow to build.

**What we added:** A `.dockerignore` file that tells Docker to skip unnecessary files:

```
.git
__pycache__
.venv
store/
tests/
Demo.gif
Demo.mp4
.env
```

Key savings:
- `Demo.mp4` — large video file, not needed inside the container
- `store/` — ChromaDB is rebuilt fresh on every startup anyway
- `.git/` — entire git history serves no purpose inside the container
- `__pycache__` — compiled Python files that get regenerated automatically
