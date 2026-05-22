import os
import io
import re
from typing import Optional
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

class RAGPipeline:
    def __init__(self):
        self.chroma = chromadb.PersistentClient(path="./chroma_db")
        self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.chroma.get_or_create_collection(
            name="banking_docs",
            embedding_function=self.embed_fn
        )
        self._load_default_data()

    def _load_default_data(self):
        """Load synthetic banking FAQ data on first run."""
        if self.collection.count() > 0:
            return
        print("Loading default banking FAQ data...")
        faq_text = open("data/banking_faq.txt").read()
        self._chunk_and_store(faq_text, "banking_faq.txt")
        print(f"Loaded {self.collection.count()} chunks.")

    def _extract_text(self, content: bytes, filename: str) -> str:
        if filename.endswith(".txt"):
            return content.decode("utf-8", errors="ignore")
        elif filename.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return ""

    def _chunk_text(self, text: str) -> list[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append(chunk)
            i += CHUNK_SIZE - CHUNK_OVERLAP
        return [c for c in chunks if len(c.strip()) > 50]

    def _chunk_and_store(self, text: str, source: str):
        chunks = self._chunk_text(text)
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metas = [{"source": source, "chunk": i} for i in range(len(chunks))]
        # Add in batches to avoid limits
        batch = 100
        for start in range(0, len(chunks), batch):
            self.collection.add(
                documents=chunks[start:start+batch],
                ids=ids[start:start+batch],
                metadatas=metas[start:start+batch]
            )
        return len(chunks)

    def ingest(self, content: bytes, filename: str) -> dict:
        text = self._extract_text(content, filename)
        if not text.strip():
            return {"chunks": 0, "error": "No text extracted"}
        n = self._chunk_and_store(text, filename)
        return {"chunks": n}

    def get_doc_count(self) -> int:
        return self.collection.count()

    def query(self, question: str, history: list[dict]) -> tuple[str, list[str]]:
        # Build context-aware query using last user message + current
        query_text = question
        if history:
            last = [m["content"] for m in history[-4:] if m["role"] == "user"]
            if last:
                query_text = " ".join(last[-1:]) + " " + question

        results = self.collection.query(query_text=query_text, n_results=TOP_K)
        docs = results["documents"][0] if results["documents"] else []
        sources = list({m["source"] for m in results["metadatas"][0]}) if results["metadatas"] else []
        context = "\n\n".join(docs)

        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful banking support assistant for a fintech company.
Answer customer questions about loans, credit cards, banking policies, and FAQs.
Base your answers on the retrieved context below. If the answer isn't in the context, 
say so politely and suggest contacting customer support.
Be concise, clear, and professional.

Retrieved context:
---
{context}
---"""
            }
        ] + history + [{"role": "user", "content": question}]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        reply = response.choices[0].message.content
        return reply, sources 