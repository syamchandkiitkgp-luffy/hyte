# Deploy HyTE on Streamlit Community Cloud

## 1) Prepare your GitHub repo

1. Push this project to a GitHub repository.
2. Ensure these files exist in the repo root:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`

## 2) Create the Streamlit app

1. Go to https://share.streamlit.io
2. Click **New app**.
3. Select your repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

## 3) Add secrets in Streamlit Cloud

In your app settings, open **Secrets** and add at least:

```toml
GEMINI_API_KEY = "your_gemini_key"
```

Optional secrets:

```toml
# Multiple keys for failover
GEMINI_API_KEYS = "key1,key2,key3"

# Optional Phoenix tracing (disabled by default)
ENABLE_PHOENIX = "false"
PHOENIX_COLLECTOR_ENDPOINT = "https://your-phoenix-endpoint/v1/traces"

# Optional Neo4j remote instance
NEO4J_URI = "neo4j+s://your-host:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
```

## 4) Cloud behavior you should expect

- If Neo4j credentials are not provided, HyTE falls back to CSV-based metadata retrieval from `Data_Dictionary/data_dictionary_enriched.csv`.
- Phoenix observability is disabled unless `ENABLE_PHOENIX=true`.
- Generated artifacts are stored in the app filesystem for the running instance; they are not durable across app restarts.

## 5) Common deployment issues

- **Gemini key missing**: The app will show a warning and LLM calls will fail.
- **Large dependency build times**: First deploy can take several minutes.
- **Neo4j connectivity**: Localhost Neo4j does not work on Streamlit Cloud. Use a hosted Neo4j URI.

## 6) Recommended production hardening

1. Move generated artifacts to cloud storage (S3, GCS, Azure Blob).
2. Use a managed vector DB instead of local Chroma persistence.
3. Add request throttling and per-session execution limits.
4. Add authentication for public deployments.
