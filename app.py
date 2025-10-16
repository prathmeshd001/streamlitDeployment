import os, re, json, math, datetime as dt
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Silence sklearn version mismatch warnings (optional)
try:
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# Optional imports for URL mode (if missing, we'll handle gracefully)
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

# Optional sentiment; if missing, we fallback to means
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

MODEL_XGB_PATH = "model_xgb_smote_best.joblib"
MODEL_RF_PATH  = "model_random_forest_balanced.joblib"
MODEL_LR_PATH  = "model_logreg_balanced.joblib"

COLS_PATH  = "feature_columns.json"
MEAN_PATH  = "feature_means.json"
DATASET    = "OnlineNewsPopularity.csv"

# bump this if you change code and want Streamlit to reload cached resources
CACHE_BUSTER = "v3"

st.set_page_config(page_title="News Popularity Predictor", page_icon="📰", layout="wide")
st.title("📰 News Popularity Predictor")
st.caption("XGBoost (SMOTE, tuned) + optional RandomForest / LogisticRegression if present")

# ---------- Load artifacts ----------
@st.cache_resource(show_spinner=False)
def load_artifacts(cache_buster="v1"):
    # Core artifacts
    model_xgb = joblib.load(MODEL_XGB_PATH)
    with open(COLS_PATH) as f: feature_columns = json.load(f)
    with open(MEAN_PATH) as f: feature_means = json.load(f)

    # Optional models (load if files exist)
    model_rf = joblib.load(MODEL_RF_PATH) if os.path.exists(MODEL_RF_PATH) else None
    model_lr = joblib.load(MODEL_LR_PATH) if os.path.exists(MODEL_LR_PATH) else None

    models = {"XGBoost (SMOTE, tuned)": model_xgb}
    if model_rf is not None: models["Random Forest (balanced)"] = model_rf
    if model_lr is not None: models["Logistic Regression (balanced)"] = model_lr

    return models, feature_columns, feature_means

MODELS, FEATURE_COLUMNS, FEATURE_MEANS = load_artifacts(cache_buster=CACHE_BUSTER)

def sanitize_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Align incoming DF to model columns; drop url/timedelta/popular/shares;
    fill only missing values with means. Also strip headers & coerce numerics.
    """
    df = df_raw.copy()

    # 1) Normalize headers so they match FEATURE_COLUMNS
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 2) Drop known non-feature columns
    drop_cols = [c for c in df.columns if c.lower() in {"url","timedelta","popular","shares"}]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 3) Ensure all expected columns exist (create missing as NaN)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # 4) Reorder to FEATURE_COLUMNS exactly
    df = df[FEATURE_COLUMNS]

    # 5) Coerce to numeric then fill NaNs with means (preserves real values)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(pd.Series(FEATURE_MEANS))

    return df

def predict_with_model(model, X: pd.DataFrame):
    """Return pred (0/1) and proba for POPULAR class if model supports it."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:,1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        proba = 1 / (1 + np.exp(-s))
    else:
        proba = np.full(shape=(len(X),), fill_value=np.nan)
    pred = (proba >= 0.5).astype(int)
    return pred, proba

# ---------- Feature importance helpers ----------
def xgb_gain_importances_series(model, feature_columns):
    try:
        booster = model.get_booster()
        raw = booster.get_score(importance_type="gain") or {}
        mapped = {}
        for k, v in raw.items():
            if k in feature_columns:
                mapped[k] = float(v)
            elif k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(feature_columns):
                    mapped[feature_columns[idx]] = float(v)
        s = pd.Series(mapped, dtype="float64")
        if not s.empty:
            return s
    except Exception:
        pass
    try:
        imp = np.asarray(model.feature_importances_, dtype=float)
        if imp.size == len(feature_columns):
            return pd.Series(imp, index=feature_columns, dtype="float64")
    except Exception:
        pass
    return pd.Series(dtype="float64")

def rf_importances_series(model, feature_columns):
    try:
        imp = np.asarray(model.feature_importances_, dtype=float)
        if len(imp) == len(feature_columns):
            return pd.Series(imp, index=feature_columns, dtype="float64")
    except Exception:
        pass
    return pd.Series(dtype="float64")

def lr_abs_coef_series(model, feature_columns):
    try:
        lr = model
        if hasattr(model, "named_steps"):
            for key in ["lr", "logisticregression"]:
                if key in model.named_steps:
                    lr = model.named_steps[key]
                    break
        coef = getattr(lr, "coef_", None)
        if coef is not None:
            arr = np.abs(np.ravel(coef)).astype(float)
            if len(arr) == len(feature_columns):
                return pd.Series(arr, index=feature_columns, dtype="float64")
    except Exception:
        pass
    return pd.Series(dtype="float64")

# ---------- URL feature extraction (beta) ----------
BASIC_STOPWORDS = {"a","an","the","and","or","but","if","while","of","to","in","on","for","by","with",
                   "is","are","was","were","be","been","being","it","its","as","at","from","that","this",
                   "these","those","we","you","he","she","they","them","his","her","their","our","i","me",
                   "my","mine","your","yours"}

def _tokenize(text: str):
    import re as _re
    return _re.findall(r"\\b\\w+\\b", (text or "").lower())

def _count_nonstop(tokens):
    return sum(1 for t in tokens if t not in BASIC_STOPWORDS)

def _get_week_flags(date_obj):
    name_map = {
        0: "weekday_is_monday", 1: "weekday_is_tuesday", 2: "weekday_is_wednesday",
        3: "weekday_is_thursday", 4: "weekday_is_friday", 5: "weekday_is_saturday",
        6: "weekday_is_sunday",
    }
    flags = {}
    wd = date_obj.weekday()
    for i, col in name_map.items():
        if col in FEATURE_COLUMNS:
            flags[col] = 1 if wd == i else 0
    if "is_weekend" in FEATURE_COLUMNS:
        flags["is_weekend"] = 1 if wd in (5,6) else 0
    return flags

def _guess_channel_flags(url, title, text):
    channel_cols = [c for c in FEATURE_COLUMNS if c.startswith("data_channel_is_")]
    flags = {c: 0 for c in channel_cols}
    hay = " ".join([(url or "").lower(), (title or "").lower(), (text or "").lower()])
    keywords = {
        "data_channel_is_entertainment": ["entertain","movie","film","music","celebrity","bollywood","hollywood","show"],
        "data_channel_is_tech": ["tech","software","gadget","ai","ml","device","startup","engineering"],
        "data_channel_is_bus": ["business","market","stock","finance","economy","trade","company"],
        "data_channel_is_world": ["world","international","global","war","country","diplomacy"],
        "data_channel_is_socmed": ["twitter","facebook","instagram","whatsapp","social media","tiktok","reddit"],
        "data_channel_is_lifestyle": ["lifestyle","health","travel","food","fitness","fashion"],
    }
    for col, words in keywords.items():
        if col in flags and any(w in hay for w in words):
            flags[col] = 1
            break
    return flags

def extract_features_from_url(url: str):
    if requests is None or BeautifulSoup is None:
        raise RuntimeError("URL mode requires 'requests' and 'beautifulsoup4'. Install them to use this feature.")

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "lxml")

    title = (soup.title.string or "").strip() if soup.title else ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip() or title

    # Article text from <p>
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)
    if len(text) < 300:
        meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
        if meta_desc and meta_desc.get("content"):
            text += " " + meta_desc["content"]

    t_tokens = _tokenize(text)
    title_tokens = _tokenize(title)

    n_tokens_content = len(t_tokens)
    n_tokens_title   = len(title_tokens)
    n_unique_tokens  = (len(set(t_tokens)) / max(1, n_tokens_content)) if n_tokens_content else 0.0
    n_non_stop_words = (_count_nonstop(t_tokens) / max(1, n_tokens_content)) if n_tokens_content else 0.0
    n_non_stop_unique_tokens = (len({t for t in t_tokens if t not in BASIC_STOPWORDS}) / max(1, n_tokens_content)) if n_tokens_content else 0.0

    anchors = soup.find_all("a")
    num_hrefs = len(anchors)
    domain = urlparse(url).netloc
    num_self_hrefs = sum(1 for a in anchors if a.get("href") and urlparse(a.get("href")).netloc == domain)

    num_imgs = len(soup.find_all("img"))
    num_videos = len(soup.find_all("video")) + sum(1 for f in soup.find_all("iframe") if f.get("src") and any(k in f["src"].lower() for k in ["youtube","vimeo"]))

    # Sentiment proxies (optional)
    if TextBlob is not None and text:
        gpol = float(TextBlob(text[:100000]).polarity)
        gsub = float(TextBlob(text[:100000]).subjectivity)
    else:
        gpol = FEATURE_MEANS.get("global_sentiment_polarity", 0.0)
        gsub = FEATURE_MEANS.get("global_subjectivity", 0.5)

    if TextBlob is not None and title:
        tpol = float(TextBlob(title).polarity)
        tsub = float(TextBlob(title).subjectivity)
    else:
        tpol = FEATURE_MEANS.get("title_sentiment_polarity", 0.0)
        tsub = float(FEATURE_MEANS.get("title_subjectivity", 0.5))

    # Publish date estimate -> weekday flags
    pub = soup.find("meta", attrs={"property": "article:published_time"}) or soup.find("meta", attrs={"name": "pubdate"})
    if pub and pub.get("content"):
        try:
            pub_dt = dt.datetime.fromisoformat(pub["content"].replace("Z","+00:00")).date()
        except Exception:
            pub_dt = dt.date.today()
    else:
        pub_dt = dt.date.today()

    week_flags = _get_week_flags(pub_dt)
    channel_flags = _guess_channel_flags(url, title, text)

    # Start from means, then overwrite what we computed
    row = {col: FEATURE_MEANS[col] for col in FEATURE_COLUMNS}

    def set_if(col, val):
        if col in row: row[col] = val

    set_if("n_tokens_title", n_tokens_title)
    set_if("n_tokens_content", n_tokens_content)
    set_if("n_unique_tokens", n_unique_tokens)
    set_if("n_non_stop_words", n_non_stop_words)
    set_if("n_non_stop_unique_tokens", n_non_stop_unique_tokens)
    set_if("num_hrefs", num_hrefs)
    set_if("num_self_hrefs", num_self_hrefs)
    set_if("num_imgs", num_imgs)
    set_if("num_videos", num_videos)
    set_if("global_subjectivity", gsub)
    set_if("global_sentiment_polarity", gpol)
    set_if("title_subjectivity", tsub)
    set_if("title_sentiment_polarity", tpol)

    if "img_vid_sum" in row:
        row["img_vid_sum"] = row.get("num_imgs",0) + row.get("num_videos",0)
    if "title_density" in row:
        row["title_density"] = (n_tokens_title / max(1, n_tokens_content)) if n_tokens_content else 0.0

    for k,v in {**week_flags, **channel_flags}.items():
        if k in row: row[k] = v

    X_new = pd.DataFrame([[row[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    meta = {"title": title[:160], "pub_date": str(pub_dt), "url": url,
            "tokens_title": n_tokens_title, "tokens_content": n_tokens_content,
            "num_imgs": num_imgs, "num_videos": num_videos, "num_hrefs": num_hrefs}
    return X_new, meta

# ---------- UI ----------
with st.sidebar:
    mode = st.radio("Input mode", ["Dataset Row", "Upload CSV", "Predict from URL"], index=0)
    model_names = list(MODELS.keys())
    compare_all = st.checkbox("Compare all available models", value=False)
    if not compare_all:
        chosen_model_name = st.selectbox("Model", model_names, index=0)
    else:
        chosen_model_name = None
    st.markdown("---")

# --- Dataset Row mode ---
if mode == "Dataset Row":
    if os.path.exists(DATASET):
        df_up = pd.read_csv(DATASET, low_memory=False)
        df_src = sanitize_df(df_up.copy())
        idx = st.slider("Row index", 0, len(df_src)-1, 0, 1)
        row = df_src.iloc[[idx]]
        st.write("**Row preview (features shown):**")
        st.dataframe(row)

        if compare_all:
            res = []
            for name, mdl in MODELS.items():
                p, proba = predict_with_model(mdl, row)
                res.append((name, int(p[0]), float(proba[0])))
            out = pd.DataFrame(res, columns=["model","pred_popular","prob_popular"]).sort_values("prob_popular", ascending=False)
            st.dataframe(out)
        else:
            mdl = MODELS[chosen_model_name]
            p, proba = predict_with_model(mdl, row)
            st.success(f"{chosen_model_name}: **{'POPULAR' if p[0]==1 else 'NOT POPULAR'}**  •  Probability(popular) = {proba[0]:.3f}")

            if "XGBoost" in chosen_model_name:
                imp = xgb_gain_importances_series(mdl, FEATURE_COLUMNS).sort_values(ascending=False).head(15)
                if not imp.empty: st.write("**Top global features:**"); st.bar_chart(imp)
            elif "Random Forest" in chosen_model_name:
                imp = rf_importances_series(mdl, FEATURE_COLUMNS).sort_values(ascending=False).head(15)
                if not imp.empty: st.write("**Top global features:**"); st.bar_chart(imp)
            else:
                imp = lr_abs_coef_series(mdl, FEATURE_COLUMNS).sort_values(ascending=False).head(15)
                if not imp.empty: st.write("**Top |coef| (LogReg):**"); st.bar_chart(imp)
    else:
        st.warning(f"Dataset '{DATASET}' not found. Use 'Upload CSV' or place it next to app.py.")

# --- Upload CSV mode ---
elif mode == "Upload CSV":
    file = st.file_uploader("Upload CSV compatible with UCI schema (shares/url columns will be ignored)", type=["csv"])
    if file:
        up = pd.read_csv(file, low_memory=False)
        df_src = sanitize_df(up.copy())
        st.dataframe(df_src.head(3))
        if st.button("Predict all rows"):
            if compare_all:
                results = {}
                for name, mdl in MODELS.items():
                    pred, proba = predict_with_model(mdl, df_src)
                    results[name] = (pred, proba)
                N = min(50, len(df_src))
                disp = df_src.head(N).copy()
                for name, (pred, proba) in results.items():
                    disp[f"pred_{name}"]  = pred[:N]
                    disp[f"proba_{name}"] = np.round(proba[:N], 3)
                st.dataframe(disp)
                first_name = list(MODELS.keys())[0]
                full = df_src.copy()
                full[f"pred_{first_name}"], full[f"proba_{first_name}"] = results[first_name]
                st.download_button("Download predictions CSV (first model)", full.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
            else:
                mdl = MODELS[chosen_model_name]
                pred, proba = predict_with_model(mdl, df_src)
                out = df_src.copy()
                out["pred_popular"] = pred
                out["prob_popular"] = proba
                st.dataframe(out.head(50))
                st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

# --- Predict from URL mode ---
else:
    st.subheader("Predict from a public article URL")
    url = st.text_input("Article URL", placeholder="https://example.com/some-article")
    if st.button("Fetch & Predict") and url:
        try:
            X_new, meta = extract_features_from_url(url)
            st.write("**Extracted meta:**", meta)

            if compare_all:
                res = []
                for name, mdl in MODELS.items():
                    p, proba = predict_with_model(mdl, X_new)
                    res.append((name, int(p[0]), float(proba[0])))
                out = pd.DataFrame(res, columns=["model","pred_popular","prob_popular"]).sort_values("prob_popular", ascending=False)
                st.dataframe(out)
            else:
                mdl = MODELS[chosen_model_name]
                p, proba = predict_with_model(mdl, X_new)
                st.success(f"{chosen_model_name}: **{'POPULAR' if p[0]==1 else 'NOT POPULAR'}**  •  Probability(popular) = {proba[0]:.3f}")

                if "XGBoost" in chosen_model_name:
                    imp = xgb_gain_importances_series(mdl, FEATURE_COLUMNS).sort_values(ascending=False).head(15)
                    if not imp.empty: st.write("**Top global features:**"); st.bar_chart(imp)
                elif "Random Forest" in chosen_model_name:
                    imp = rf_importances_series(mdl, FEATURE_COLUMNS).sort_values(ascending=False).head(15)
                    if not imp.empty: st.write("**Top global features:**"); st.bar_chart(imp)
                else:
                    imp = lr_abs_coef_series(mdl, FEATURE_COLUMNS).sort_values(ascending=False).head(15)
                    if not imp.empty: st.write("**Top |coef| (LogReg):**"); st.bar_chart(imp)
        except Exception as e:
            st.error(f"Failed to extract/predict from URL: {e}")
            if requests is None or BeautifulSoup is None:
                st.info("Install dependencies:  pip install beautifulsoup4 lxml textblob")
