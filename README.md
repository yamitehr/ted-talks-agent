# TED Talk RAG Agent

This is a RAG-based agent for querying TED Talks, built with LangChain, Pinecone, and FastAPI.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Copy `.env.example` to `.env` and fill in your API keys.
    ```bash
    cp .env.example .env
    ```
    Required keys:
    - `OPENAI_API_KEY`
    - `OPENAI_BASE_URL`
    - `PINECONE_API_KEY`
    - `PINECONE_INDEX_NAME` (default: `ted-talks`)

## Ingestion

Run the ingestion script to load data into Pinecone. You can start with a subset to test.

```bash
# Ingest first 10 talks
python ingest.py --subset 10

# Ingest all talks
python ingest.py
```

Arguments:
- `--subset`: Number of talks to ingest.

## Running the API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Endpoints

- `POST /api/prompt`: Query the agent.
  ```json
  {
    "question": "What does Al Gore say about climate change?"
  }
  ```
- `GET /api/stats`: Get RAG configuration.

## RAG Hyperparameters

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| **Chunk Size** | 1000 tokens | Large enough to capture complete thoughts and context within a speech (preserving semantic meaning), but small enough to fit well within the 2048 limit. |
| **Overlap** | 200 tokens (20%) | Ensures semantic continuity across chunk boundaries so sentences aren't cut mid-thought. Stays safely under the 30% limit. |
| **Top-K** | 20 | Optimized for diversity. Since a single talk (~2500 tokens) consumes ~3-4 chunks, a lower Top-K (e.g., 5) would risk retrieving chunks from only one talk. Top-K of 20 guarantees that at least 5-6 distinct talks appear in the context, ensuring the model can successfully answer "List 3 talks" requests while staying under the limit of 30. |
