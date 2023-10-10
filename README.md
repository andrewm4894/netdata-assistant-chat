# netdata-assistant-chat

Experimenting with chat mode for Netdata Assistant

## Usage

### Env Vars

See `.env.example` for required env vars and set them in `.env` file.
or
See `.streamlit/secrets.example.toml` for required env vars and set them in `.streamlit/secrets.toml` file.

### Build Index

Run `index_build.py` step by step to build the index and store the vectors in Pinecone.

### Run Query

You can run a query using `index_query.py` notebook to play around a bit.

### Local

Run streamlit app locally using:

```bash
streamlit run app.py
```

This will open a browser window with the app running and pulling from the live Pinecone index.
