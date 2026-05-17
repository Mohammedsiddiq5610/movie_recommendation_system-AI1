# Movie Recommendation System

This repository contains a Streamlit app (`recommendation_system.py`) that provides movie recommendations using collaborative filtering.

## Run locally

1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows cmd
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run recommendation_system.py
```

Open http://localhost:8501 in your browser.

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and log in.
3. Click "New app" → connect your GitHub repo → choose branch `main` and set the file path to `recommendation_system.py`.
4. Click "Deploy". Streamlit will install packages from `requirements.txt` and run the app.

Notes
- Ensure the data files (`movies.csv`, `ratings.csv`, `links.csv`) are present in the repository root on GitHub if you want the deployed app to include the same dataset.
- `requirements.txt` lists required packages: `pandas`, `scikit-learn`, `streamlit`.

