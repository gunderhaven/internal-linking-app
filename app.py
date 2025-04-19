import os
import streamlit as st
import pandas as pd
import numpy as np
import ast

# --------------------------
# Install and Import OpenAI
# --------------------------
try:
    import openai
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "openai"], check=True)
    import openai

from collections import defaultdict
from datetime import date
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Internal Link Recommender", layout="wide")
st.title("🔗 Internal Link Recommendation Tool")

# --------------------------
# Sidebar: Settings & API Key
# --------------------------
st.sidebar.header("⚙️ Settings")
min_similarity_threshold = st.sidebar.slider("Minimum Similarity", 0.0, 1.0, 0.75)
max_outbound_per_source = st.sidebar.number_input("Max Outbound per Source", min_value=1, value=3)

# OpenAI API Key Input
st.sidebar.subheader("🔑 OpenAI API Key")
user_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
env_api_key = os.getenv("OPENAI_API_KEY")
if not user_api_key and not env_api_key:
    st.sidebar.error("Please provide your OpenAI API key to enable anchor-text suggestions.")
    st.stop()
openai.api_key = user_api_key or env_api_key

# --------------------------
# Description & Help
# --------------------------
st.markdown(
    """
This tool recommends **internal links** between your site's pages based on **semantic similarity** using page embeddings.

Upload a single CSV file containing URL data, internal links, and precomputed embeddings.
"""
)

with st.expander("🛠 How to Generate the Input CSV (using Screaming Frog)", expanded=False):
    st.markdown("""
(Instructions unchanged)...
""")

# --------------------------
# File Upload and Mapping
# --------------------------
uploaded_file = st.file_uploader("📄 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = list(df.columns)
    st.sidebar.subheader("📑 Column Mapping")
    url_col = st.sidebar.selectbox("URL column", cols)
    title_col = st.sidebar.selectbox("Title column", cols)
    links_col = st.sidebar.selectbox("Links column", cols)
    embedding_col = st.sidebar.selectbox("Embedding column", cols)
    df = df.rename(columns={url_col: "URL", title_col: "Title", links_col: "Links", embedding_col: "Embedding"})

    # Target Prioritization
    st.sidebar.subheader("📌 Target Prioritization")
    target_priority_mode = st.sidebar.selectbox("Prioritize targets by", ["None", "Low inlink count", "URL contains keyword"])
    target_keyword = ""
    if target_priority_mode == "URL contains keyword":
        target_keyword = st.sidebar.text_input("Enter target keyword (e.g. /blog/)")
    target_priority_strength = 1.0
    if target_priority_mode != "None":
        target_priority_strength = st.sidebar.slider("Target Priority Strength", 0.5, 5.0, 2.0, help="Higher values give stronger priority to targeted pages")

    # Source Prioritization
    st.sidebar.subheader("📌 Source Prioritization")
    source_priority_mode = st.sidebar.selectbox("Prioritize sources by", ["None", "URL contains keyword"])
    source_keyword = ""
    if source_priority_mode == "URL contains keyword":
        source_keyword = st.sidebar.text_input("Enter source keyword (e.g. /section/)")
    source_priority_strength = 1.0
    if source_priority_mode != "None":
        source_priority_strength = st.sidebar.slider("Source Priority Strength", 0.5, 5.0, 2.0, help="Higher values give stronger priority to source pages")

    # Run Button
    run_analysis = st.sidebar.button("Run Analysis")
    if not run_analysis:
        st.info("After mapping your columns and adjusting settings, click 'Run Analysis' in the sidebar.")
    else:
        # --------------------------
        # Main Logic
        # --------------------------
        try:
            df["Links"] = df["Links"].apply(ast.literal_eval)
            df["Embedding"] = df["Embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
            urls = df["URL"].tolist()
            titles = dict(zip(df["URL"], df["Title"]))
            embeddings = np.vstack(df["Embedding"].values)

            # Count inlinks
            inlink_counter = defaultdict(int)
            for src, links in zip(df["URL"], df["Links"]):
                for link in links:
                    if link.startswith('/'):
                        for u in urls:
                            if u.endswith(link): inlink_counter[u] += 1
                    else:
                        inlink_counter[link] += 1
            if target_priority_mode == "Low inlink count":
                st.subheader("📊 Current Inlink Distribution")
                st.dataframe(
                    pd.DataFrame({"URL": list(inlink_counter), "Inlink Count": list(inlink_counter.values())})
                    .sort_values("Inlink Count")
                )

            # Build existing links
            existing = set()
            for src, links in zip(df["URL"], df["Links"]):
                for t in links:
                    if t.startswith('/'):
                        for u in urls:
                            if u.endswith(t): existing.add((src, u))
                    else:
                        existing.add((src, t))

            sim_matrix = cosine_similarity(embeddings)

            def target_boost(tgt):
                if target_priority_mode == "Low inlink count":
                    cnt = inlink_counter.get(tgt, 0) + 1
                    return max(1.0, 1 + (target_priority_strength - 1) * (1 / np.log(cnt + 1.1)))
                if target_priority_mode == "URL contains keyword" and target_keyword.lower() in tgt.lower():
                    return target_priority_strength
                return 1.0

            def source_boost(src):
                if source_priority_mode == "URL contains keyword" and source_keyword.lower() in src.lower():
                    return source_priority_strength
                return 1.0

            outbound = defaultdict(int)
            recommendations = []
            progress = st.progress(0)
            for i, tgt in enumerate(urls):
                progress.progress((i+1)/len(urls))
                for j, src in enumerate(urls):
                    if src == tgt or (src, tgt) in existing or outbound[src] >= max_outbound_per_source:
                        continue
                    score = sim_matrix[j,i]
                    if score < min_similarity_threshold: continue
                    boosted = score * target_boost(tgt) * source_boost(src)
                    recommendations.append((src, tgt, score, boosted))
                    outbound[src] += 1
            progress.empty()

            # Build results DataFrame
            rec_df = pd.DataFrame(recommendations, columns=["SourceURL","TargetURL","Score","Boosted"]) 
            rec_df["SourceTitle"] = rec_df["SourceURL"].map(titles)
            rec_df["TargetTitle"] = rec_df["TargetURL"].map(titles)
            rec_df["Date"] = date.today().isoformat()
            rec_df["AnchorText"] = ""

            # Generate anchor text
            for idx, row in rec_df.iterrows():
                prompt = f"Suggest concise anchor text for linking '{row['SourceTitle']}' to '{row['TargetTitle']}'."
                try:
                    resp = openai.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=[
                            {"role":"system","content":"You suggest concise anchor text for hyperlinks."},
                            {"role":"user","content":prompt}
                        ], temperature=0.7
                    )
                    rec_df.at[idx, "AnchorText"] = resp.choices[0].message.content.strip()
                except Exception as err:
                    st.error(f"Anchor-text generation failed at row {idx}: {err}")

            # Display output
            if not rec_df.empty:
                st.success(f"✅ Found {len(rec_df)} link opportunities")
                display_cols = ["SourceURL","SourceTitle","TargetURL","TargetTitle","Score","Boosted","AnchorText","Date"]
                st.dataframe(rec_df.sort_values("Boosted",ascending=False)[display_cols])
                st.download_button("📥 Download CSV", rec_df.to_csv(index=False).encode(), "links.csv", "text/csv")
            else:
                st.warning("No recommendations with current settings.")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Check your file format and settings.")
