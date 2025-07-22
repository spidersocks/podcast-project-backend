# Topic Drilldown API (FastAPI)
A FastAPI backend for hierarchical topic distribution and drilldown.

## Features
- Loads data from data/news_chunks_w_umap.csv
- Two endpoints:
    - `/api/topics/ — broadest topic distribution`
    - `/api/topics/drilldown/?path=topic1,topic2 — subtopic distribution for any drill path`

## Usage
1. **Install dependencies:**
```
pip install fastapi uvicorn pandas
```

2. **Start the server:**
```
uvicorn main:app --reload
```

3. **Endpoints:**

- `GET /api/topics/`  
  → Returns broadest topics and their counts/proportions per `source_type`.

- `GET /api/topics/drilldown/?path=society,fundamental rights`  
  → Returns next-level subtopics beneath the given path.

4. **Response format:**

```json
{
  "topics": [
    {
      "label": "civil rights",
      "source_type": "news",
      "count": 25,
      "source_total": 100,
      "proportion": 0.25
    }
  ]
}
```

## Notes
- Expects `all_topics` as a list or stringified list in the CSV.
- CORS is enabled for all origins (edit in `main.py` for production).