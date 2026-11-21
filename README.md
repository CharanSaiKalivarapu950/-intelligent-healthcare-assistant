# Intelligent Healthcare Assistant

Educational portfolio project combining chest X-ray image classification, Grad-CAM visualization, and document-grounded medical knowledge retrieval (RAG).

**Safety:** This project is educational research only. It is not a medical diagnosis tool. Never use its outputs for clinical decisions, treatment, or triage.

## Features
- ResNet18 transfer-learning training on a folder-based image dataset
- Image prediction with probabilities
- Grad-CAM heatmap overlay
- PDF ingestion, extraction, chunking, and metadata
- Sentence Transformers embeddings + FAISS semantic search
- LLM answer generation through an OpenAI-compatible API
- Answers include retrieved source names and page/chunk metadata
- FastAPI API and Streamlit interface

## Dataset setup
Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Expected structure:
```text
datasets/raw/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```
Do not commit datasets. Review the dataset license and limitations.

## Setup (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
For Apple Silicon, PyTorch may use MPS. Training script detects CUDA, MPS, or CPU.

## Train
```bash
python -m backend.app.ml.train --data datasets/raw/chest_xray --epochs 5 --batch-size 16
```
Checkpoint is written to `models/pneumonia_resnet18.pth`.

## Start API
```bash
uvicorn backend.app.main:app --reload
```
Docs: http://127.0.0.1:8000/docs

## Start UI (separate terminal)
```bash
streamlit run frontend/streamlit_app.py
```

## RAG usage
1. Put PDF documents you have permission to use in `medical_documents/raw/`.
2. Start API.
3. In Streamlit, choose Knowledge Assistant and ingest PDFs, or call `POST /api/rag/ingest`.
4. Ask a question. Retrieval works without an LLM; if configured, an LLM will synthesize a grounded response. Without a configured LLM, API returns retrieved passages.

## API endpoints
- `GET /api/health`
- `GET /api/model/status`
- `POST /api/image/predict` (multipart image)
- `POST /api/rag/ingest`
- `POST /api/rag/ask` (JSON `{ "question": "...", "top_k": 4 }`)

## Docker
Build and run API:
```bash
docker build -t healthcare-assistant .
docker run --rm -p 8000:8000 healthcare-assistant
```
The container does not include your dataset or trained checkpoint. Mount these if needed.

## Project layout
```text
backend/app/
  api/       HTTP routes
  core/      settings
  ml/        training, inference, Grad-CAM
  rag/       PDF extraction, chunking, embeddings, vector search, generation
frontend/    Streamlit UI
medical_documents/raw/  input PDFs
medical_documents/index/ persisted FAISS index
models/     trained checkpoint
tests/      API tests
docs/       model card and evaluation template
```

## Responsible use and evaluation
- Split data at patient level when identifiers permit; this public dataset's directory split is not guaranteed to represent independent patients.
- Inspect class balance and image sources.
- Report recall, precision, F1, confusion matrix, ROC-AUC when applicable, not accuracy alone.
- Model confidence is not clinical certainty.
- Grad-CAM is a visualization of model sensitivity, not proof of reasoning.
- RAG answers can be wrong even with citations; verify sources.
