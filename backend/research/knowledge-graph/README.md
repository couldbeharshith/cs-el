# Knowledge graph research

`create_KG.ipynb` and `query_KG.ipynb` use the local `KG_full.py` and `utils.py` modules. Open the notebooks with this directory as their working directory and install `requirements.txt` in a dedicated notebook environment.

The `v1/` prototype builds a policy graph using Azure Document Intelligence, OpenAI, and Neo4j. Install `v1/requirements.txt` in its own environment. Run the scripts from `v1/` and configure credentials with the variables in `config.py`. `query_api_client.py` can check the Aura query endpoint. `ingest_extracted.py` reads `extracted_policy.md` from the working directory and writes an ingestion report.

Use a development graph database for notebook and ingestion runs. `CLEAR_BEFORE_INGEST` enables removal of previously ingested markdown nodes.
