```markdown
## Insurance RAG API Specification

**Endpoint:** `POST /api/v1/hackrx/run`

**Authentication:** Bearer token required in `Authorization` header

### Request Format
```json
{
  "documents": "string (URL, 10-2048 chars, https/http only, no localhost/private IPs)",
  "questions": "list[string] (1-50 items, each 3-1000 chars, no harmful content)"
}
```

### Response Format
```json
{
  "answers": "list[string] (one answer per question with decision, reasoning, and citations)"
}
```

### Supported Document Types
PDF, DOCX, XLSX, PPTX, TXT, EML, images (.png, .jpg, .gif, .webp, .heic), audio (.wav, .mp3, .aiff, .aac, .ogg, .flac <0.5MB), video (.mp4, .mpeg, .mov, .avi, .webm, .wmv, .3gpp <0.5MB)

### Processing Pipeline
1. Fetch document from URL (max 1GB configurable)
2. Classify by type/size
3. Fast path: Images/small files (<0.5MB) → direct Gemini multimodal (no chunking)
4. Large files: Extract text → PDF/DOCX/PPTX/XLSX parsing with format-specific tools
5. TikToken chunking (cl100kbase, configurable size/overlap via .env)
6. Embed chunks (llama-text-embed-v2) and upsert to Pinecone
7. Optional relevance filtering (similarity threshold configurable)
8. Query expansion (3 rephrased queries via Groq/Gemini)
9. Vector search Pinecone for top_k results per expanded query
10. Deduplicate and rerank chunks (BGE/Cohere)
11. Final LLM call with top_n chunks as context (Gemini/Groq)
12. Return consolidated answer

### Configuration
All settings in .env: size limits, thresholds, top_k/top_n, model choices, chunking params, API key rotation (primary/fallback pools for Gemini/Groq/Pinecone)

### Error Response
```json
{
  "error_type": "string",
  "message": "string",
  "details": "object|null",
  "timestamp": "ISO 8601",
  "request_id": "UUID"
}
```
