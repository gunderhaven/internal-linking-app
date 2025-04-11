import streamlit as st
import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from datetime import date
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Internal Link Recommender", layout="wide")
st.title("🔗 Internal Link Recommendation Tool")

# --------------------------
# 📘 App Description
# --------------------------
st.markdown("""
This tool recommends **internal links** between your pages based on **semantic similarity**.

### 📥 Input Format
Upload a **single CSV file** with the following columns:

- **URL** – The full URL of the page  
- **Title** – The page title (for context)  
- **Links** – A stringified list of internal links from this page (e.g. "['/page-a', '/page-b']")  
- **Embedding** – A stringified list of numbers representing the page’s content embedding (e.g. "[0.12, -0.03, ...]")

Embeddings should be generated beforehand using a model like OpenAI’s `text-embedding-ada-002`.

The app uses those embeddings to find topically relevant pages that aren't already linked together, then suggests high-quality internal link opportunities.
""")

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("📄 Upload your CSV", type="csv")

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.header("⚙️ Settings")
min_similarity_threshold = st.sidebar.slider("Minimum Similarity", 0.0, 1.0, 0.75)
max_outbound_per_source = st.sidebar.number_input("Max Outbound per Source", min_value=1, value=3)

MAX_INBOUND = 99999  # disable inbound limit

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Parse stringified lists
    df["Links"] = df["Links"].apply(ast.literal_eval)
    df["Embedding"] = df["Embedding"].apply(lambda x: np.array(ast.literal_eval(x)))

    urls = df["URL"].tolist()
    titles = dict(zip(df["URL"], df["Title"]))
    embeddings = np.vstack(df["Embedding"].values)

    # Build inlink count
    inlink_counter = defaultdict(int)
    for links in df["Links"]:
        for link in links:
            inlink_counter[link] += 1

    # Build existing link set
    existing_links = set()
    for source, links in zip(df["URL"], df["Links"]):
        for target in links:
            existing_links.add((source, target))

    # Compute pairwise similarity
    sim_matrix = cosine_similarity(embeddings)

    outbound_counter = defaultdict(int)
    inbound_counter = defaultdict(int)
    recommendations = []

    for i, target_url in enumerate(urls):
        if inlink_counter.get(target_url, 0) >= MAX_INBOUND:
            continue

        candidate_sims = []
        for j, source_url in enumerate(urls):
            if source_url == target_url:
                continue
            if (source_url, target_url) in existing_links:
                continue
            if outbound_counter[source_url] >= max_outbound_per_source:
                continue

            sim_score = sim_matrix[j, i]
            if sim_score < min_similarity_threshold:
                continue

            candidate_sims.append((source_url, target_url, sim_score))

        candidate_sims.sort(key=lambda x: x[2], reverse=True)

        for source_url, target_url, score in candidate_sims[:max_outbound_per_source]:
            outbound_counter[source_url] += 1
            inbound_counter[target_url] += 1
            recommendations.append({
                "SourceURL": source_url,
                "SourceTitle": titles.get(source_url, ""),
                "TargetURL": target_url,
                "TargetTitle": titles.get(target_url, ""),
                "SimilarityScore": round(score, 4),
                "Status": "",
                "Date": date.today().isoformat(),
                "Notes": ""
            })

    recommendations_df = pd.DataFrame(recommendations)

    st.success(f"✅ Found {len(recommendations_df)} internal link opportunities")
    st.dataframe(recommendations_df)

    csv = recommendations_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="recommended_internal_links.csv", mime="text/csv")
