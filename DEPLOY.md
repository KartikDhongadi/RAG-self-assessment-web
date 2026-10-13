# Deploying the web app for free

Target: a public URL anyone can open, at zero cost.

| Piece | Service | Cost |
|---|---|---|
| Web hosting | Streamlit Community Cloud | Free |
| LLM inference | Groq API | Free tier, no credit card, never billed |
| Embeddings | Chroma's bundled ONNX all-MiniLM-L6-v2 | Runs in-process |
| Vector store | ChromaDB, built from `demo_docs/` on startup | Free |

---

## 1. Get a Groq API key

1. Sign up at <https://console.groq.com> (no card needed).
2. **API Keys → Create API Key**, copy it (starts with `gsk_`). It is shown once.
3. Never commit it. `.streamlit/secrets.toml` is already in `.gitignore`.

The default model is `openai/gpt-oss-120b`. Which models a key can use varies
by account. If you get a "model isn't available" error, list your key's models:

```bash
python -c "import tomllib,groq; k=tomllib.load(open('.streamlit/secrets.toml','rb'))['GROQ_API_KEY']; print([m.id for m in groq.Groq(api_key=k).models.list().data])"
```

then set `GROQ_MODEL = "<one of them>"` in your secrets.

Each question costs about 3 Groq requests (batched grading), so the free tier
handles a few hundred questions a day. Going over returns a rate-limit error,
never a charge.

---

## 2. Run it on your laptop first

Use Python 3.11 or 3.12 (3.14 is too new for some of these packages).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p .streamlit
echo 'GROQ_API_KEY = "gsk_your_key_here"' > .streamlit/secrets.toml

streamlit run app.py
```

The first run downloads the small embedding model (~80 MB) and indexes the
15 PDFs in `demo_docs/`, so give it a minute.

To use your local Ollama instead of Groq (no key needed):

```bash
pip install -r requirements-local.txt
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2 streamlit run app.py
```

---

## 3. Push to GitHub

```bash
git add app.py rag_core.py requirements.txt requirements-local.txt \
        DEPLOY.md README.md .gitignore demo_docs/
git status        # check secrets.toml is NOT listed
git commit -m "Add Streamlit web UI, Groq provider, batched grading, demo corpus"
git push
```

---

## 4. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → repo `KartikDhongadi/RAG-self-assessment-web`,
   branch `main`, main file `app.py`.
3. **Advanced settings**: choose Python 3.12, and paste into **Secrets**:

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

4. Deploy. The first boot takes a few minutes (installing packages, indexing).

Your URL will look like `https://<app-name>.streamlit.app`. Add it to the
README and your CV.

---

## Good to know

- **Sleeping:** free apps sleep after a period without visitors. The next
  visitor clicks a button to wake it, and the index rebuilds in under a minute.
- **Memory:** free apps get roughly 1–2.7 GB. Keep `demo_docs/` small, and
  don't add `sentence-transformers`/PyTorch to `requirements.txt`.
- **Public key usage:** anyone using the app spends your Groq quota. It's
  capped, not billed, so the worst case is rate-limit errors.
- **Uploads:** visitor uploads are indexed in memory for that visit only
  (max 5 files, 10 MB each) and are never saved or committed.
- **Everything in `demo_docs/` is public**, both in the repo and through the app.
