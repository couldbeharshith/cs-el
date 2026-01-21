---
title: Insurance Agent
emoji: 🐨
colorFrom: red
colorTo: red
sdk: docker
pinned: false
short_description: API
---

# Insurance RAG API

A high-accuracy RAG-based API for processing queries from documents with URL-based document processing, Pinecone vector storage, Reranking and multiquery expansion implementation
---

## Features

- **Document Processing**: Supports PDF, DOCX, TXT, and EML files from URLs
- **PyMuPDF Integration**: Fast PDF processing with multiprocessing
- **Markdown Extraction**: Advanced PDF to markdown conversion with multiprocessing for better structure preservation
- **TikToken Chunking**: Token-aware text chunking for better accuracy
- **Vector Search**: Pinecone-powered semantic search
- **LLM Integration**: Google Gemini for intelligent responses
- **Relevance Filtering**: Pre-filters irrelevant queries to save costs (optional)
- **Concurrent Processing**: Handles multiple questions in parallel using asyncio

---

## API Endpoints

### POST `/api/v1/hackrx/run`

Process insurance policy documents and answer questions.

**Request Body:**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is covered under this policy?",
    "What are the exclusions?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "This policy covers medical expenses up to ₹5 lakhs (Page 3, Line 15)...",
    "Exclusions include pre-existing conditions (Page 8, Line 22)..."
  ]
}
```

### GET `/health`

Health check endpoint for monitoring.

---

## Supported Document Types

- PDF files (.pdf)
- Email files (.eml)
- Microsoft Word documents (.docx)
- Microsoft Excel documents (.xlsx)
- Microsoft Powerpoint documents (.pptx)
- Plain text files (.txt including code files like .html, .js, .py, .xlm, etc.)
- Image files (.png, .jpg, .bmp, .gif, .webp, .heic, .heif)
- Audio files (.wav, .mp3, .aiff, .aac, .ogg, .flac) `under 0.5 MB`
- Video files (.mp4, .mpeg, .mov, .avi, .x-flv, .webm, .wmv, .3gpp) `under 0.5 MB`

---

# Steps of processing (processing pipeline)

1. **Receive request**  
   FastAPI accepts incoming POST requests at the `/api/v1/hackrx/run` endpoint.

2. **Fetch document**  
   Download the document from the provided URL if it’s under the size limit (default `1GB`, configurable via `.env`) and not a `.zip` or `.bin`.

3. **Classify document**  
   Classify the file by type and size to determine the processing path.

4. **Direct multimodal handling (fast path)**  
   If the file is an image, or is ≤ `0.5MB` (configurable) and is a `pdf`, `txt`/`eml`, or `audio`/`video`, send it directly to Gemini’s multimodal API to answer the query without chunking, embedding, or retrieval — this significantly reduces latency.

5. **Process PDF**  
   Extract text and tables from PDFs using `pymupdf4LLM` and convert output to `Markdown` If pdf is ≤ 300 pages else we process it into plain text to save up on processing time. Processing is parallelized by batching pages across worker processes if file is large enough.

6. **Process DOCX**  
   Extract content from `.docx` files using `pymupdf` and convert to Markdown.

7. **Process PPTX**  
   Extract slides and text from `.pptx` via `python-pptx` and apply Tesseract OCR where needed; output is converted to Markdown to preserve structure of text and tables.

8. **Process XLSX**  
   Parse Excel workbooks with `openpyxl`, exporting each worksheet’s contents to Markdown table format.

9. **Chunking**  
   Split extracted text into chunks using `tiktoken`(`cl100kbase`) with chunk size and overlap defined in `.env`.

10. **Embed & store**  
    Upsert records with plain text of each chunk (`llama-text-embed-v2`) to pinecone in batches to reduce latency and store vectors along plain-text chunks in Pinecone.

11. **Relevance filtering (optional)**  
    If enabled in `.env`, compute similarity between the query embedding and chunk embeddings. Mark queries as irrelevant when the top similarity is below the configured threshold (`note`: disable for multilingual tasks). Irrelevant queries are not processed further and return a suitable API response.

12. **Query expansion (multi-query)**  
    Expand each incoming query into three rephrased queries using an LLM (primary: Groq, fallback: Gemini) to improve retrieval recall.

13. **Retrieve & deduplicate**  
    Vector-query Pinecone for the `top_k` (configurable) results for each rephrased query, merge results, and deduplicate candidate chunks.

14. **Rerank**  
    Rerank the deduplicated chunks against the original query (primary: BGE (`bge-reranker-v2-m3`), fallback: cohere (`rerank-v3.5`)) and select the `top_n` chunks (configurable).

15. **Final LLM call**  
    Send the `top_n` chunks as context to the final LLM (primary: Gemini, fallback: Groq) using prompt templates in `prompts.json`. We also provide a tool/function for the LLM to make external API calls when needed for specific queries.

16. **Return & store response**  
    Collect the final LLM output, store it, and return the consolidated answer to the API caller.

---
## Notes & operational details

- **Configuration:** Most global settings (size limits, thresholds, `top_k`, `top_n`, model preferences, etc.) are configurable in `.env`.  
- **API key rotation:** We rotate API keys across multiple Gemini, Groq, and Pinecone keys (primary pool, then secondary) to distribute load and avoid free-tier rate limits.  
- **Data classes:** All data classes we use are stores in `models.py`.
- **Prompt management:** All prompts used live in a central json file `prompts.json`.
- **Repo history:** The repo was originally hosted on Hugging Face: `https://huggingface.com/couldbeharshith/insurance-agent`. We migrated to GitHub for convenience; prior commits and the working HF endpoint are still available there.  
- **Live endpoint (Hugging Face Spaces):**  
  `https://couldbeharshith-insurance-agent.hf.space/api/v1/hackrx/run`
- **API Keys:** all of our free API keys are in the .env. *If you want to use a single API key, just copy paste the SAME API key for all API key variables in `.env`*


---