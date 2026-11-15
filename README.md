# MTS EnergyTool AI Consultant — Stage 2

## Purpose
A web-based AI consultant for MTS EnergyTool. A visitor asks a question, receives an answer grounded in the supplied MTS knowledge base, and can submit a contact request for follow-up by an MTS specialist.

## Architecture
- **Web UI:** HTML, CSS and JavaScript served by FastAPI.
- **API:** FastAPI endpoints for health, chat, conversation history and customer leads.
- **Retrieval:** heading-aware Markdown chunking, TF-IDF lexical retrieval and semantic retrieval; results are combined using Reciprocal Rank Fusion (RRF).
- **Generation:** Ollama chat model receives retrieved evidence and conversation context to produce a grounded answer.
- **Embeddings:** Ollama embedding model creates vectors for knowledge chunks and user questions.
- **Persistence:** SQLite via SQLAlchemy stores sessions, messages and submitted leads.

## Requirements
- Docker Engine / Docker Desktop with Docker Compose.
- Ollama installed and running on the host (or an Ollama endpoint reachable from the container).
- Ollama models pulled before startup:
  - `gemma3:1b`
  - `nomic-embed-text`
- The authorized MTS knowledge-base archive supplied by the organizers.

## Knowledge base placement
The organizer-provided archive is intentionally **not included** in this submission. Do not redistribute it.
Place the authorized file at:

`local_data/knowledge-base.zip`

The application currently expects this exact path. The archive must contain the Markdown knowledge base in the structure expected by `kb_core.py`.

## Configuration
Copy `.env.example` to `.env` and adjust values if necessary:
- `OLLAMA_BASE_URL`: Ollama API base URL. With Docker Desktop on Windows/macOS, the default uses `host.docker.internal`.
- `LLM_MODEL`: Ollama chat model name.
- `EMB_MODEL`: Ollama embedding model name.
- `TEMPERATURE`: generation temperature.
- `MAX_TOKENS`: maximum generation tokens.

No API keys are required for the default local Ollama setup. Never put real credentials in `.env.example` or the submission archive.

## One-command launch
From the project root, after placing the authorized knowledge-base archive and starting Ollama with both models available:

```bash
docker compose up --build
```

Open `http://localhost:8000`.

Stop the application with:

```bash
docker compose down
```

The SQLite database is stored in the named Docker volume `mts_energytool_data` at `/app/data` only if the application database path is configured to use that directory. **Current code uses `sqlite:///./mts_energytool.db` relative to the container working directory**, so verify persistence behavior before production use. `docker compose down -v` deletes the named volume.

## Local launch without Docker
Create and activate a Python 3.11 virtual environment, install dependencies, ensure Ollama and both models are available, then run:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Main scenario test order
1. Confirm Ollama is reachable and both models are installed.
2. Confirm `local_data/knowledge-base.zip` exists.
3. Start the app and open `http://localhost:8000`.
4. Ask a question about MTS EnergyTool in Russian.
5. Confirm an answer and source information are displayed.
6. Open the MTS specialist contact form.
7. Submit a test lead using fictional details.
8. Confirm the UI reports success and verify the lead is present in the SQLite `leads` table.

Health endpoint: `GET /health` (if exposed by the current application version). API documentation: `/docs`.

## Dependencies and ports
- Python 3.11
- FastAPI / Uvicorn, Pydantic, SQLAlchemy, HTTPX, NumPy, scikit-learn
- Web application: TCP port `8000`
- Ollama API: TCP port `11434` (provided by the host or external Ollama server)

## Known limitations / items to verify
- The MTS knowledge-base archive is an external organizer-provided prerequisite and is not bundled.
- Docker deployment has been scaffolded but must be tested on a clean machine with Docker Desktop and Ollama; do not describe it as validated until tested.
- Docker Compose directs SQLite to the named data volume. Verify persistence after a container restart during deployment testing.
- Ollama must be available before app startup because the application loads the knowledge base and creates embeddings during FastAPI lifespan startup.
- The project has been tested locally for the chat response and successful lead submission, including a SQLite record. This does not constitute a clean-install or Docker test.
- Add the final repository URL and deployed demo URL to this README/presentation if available.

## Security and submission hygiene
- Do not include the organizer's knowledge-base archive.
- Do not include `.env`, working keys, passwords, personal data or a live SQLite database.
- Use fictional data for screenshots and demonstrations.
