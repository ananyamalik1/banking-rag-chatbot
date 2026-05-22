# banking-rag-chatbot
AI banking chatbot with RAG using FastAPI and ChromaDB. Handles loans, savings, credit cards, and document uploads.
# 🏦 SynthBank AI Support Chatbot

An AI-powered banking support chatbot built with RAG (Retrieval-Augmented Generation), FastAPI, ChromaDB, and OpenAI.

## Architecture

```
User → Frontend (HTML/JS)
         ↓ POST /chat
       FastAPI Backend
         ↓
       RAG Pipeline
       ├── ChromaDB (vector store)
       ├── OpenAI Embeddings (text-embedding-3-small)
       └── GPT-3.5-turbo (response generation)
```

**Flow:**
1. User asks a question
2. Question is embedded using OpenAI embeddings
3. Top 4 similar chunks are retrieved from ChromaDB
4. Retrieved context + chat history is sent to GPT-3.5
5. Response returned with source attribution

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | text-embedding-3-small |
| Vector DB | ChromaDB (local persistent) |
| Backend | FastAPI + Python |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Render (free tier) |

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/banking-chatbot
cd banking-chatbot/backend
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

### 3. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open the frontend

Open `frontend/index.html` in your browser. That's it!

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + doc count |
| POST | `/chat` | Send a message, get AI reply |
| POST | `/upload` | Upload PDF or TXT document |

### POST /chat

```json
Request:  { "message": "What is the personal loan interest rate?", "session_id": "optional" }
Response: { "reply": "...", "session_id": "abc123", "sources": ["banking_faq.txt"] }
```

### POST /upload

```
Form-data: file = <PDF or TXT>
Response:  { "message": "Ingested 'doc.pdf'", "chunks": 42 }
```

## Deployment (Render — Free)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo, set root to `backend/`
4. Add environment variable: `OPENAI_API_KEY`
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Deploy! Update `API` variable in `frontend/index.html` with your Render URL.

## Project Structure

```
banking-chatbot/
├── backend/
│   ├── main.py          ← FastAPI app, API routes
│   ├── rag.py           ← RAG pipeline (chunking, embedding, retrieval)
│   └── requirements.txt
├── frontend/
│   └── index.html       ← Chat UI
├── data/
│   └── banking_faq.txt  ← Default knowledge base
└── README.md
```

## Features

- ✅ RAG pipeline with ChromaDB vector store
- ✅ Multi-turn conversation with session memory
- ✅ PDF and TXT document upload + ingestion
- ✅ Source attribution in responses
- ✅ Typing indicator
- ✅ Quick suggestion chips
- ✅ REST API (POST /chat, POST /upload, GET /health)
- ✅ Deployable to Render free tier
