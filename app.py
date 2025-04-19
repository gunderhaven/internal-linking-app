import os
import streamlit as st
import pandas as pd
import numpy as np
import ast

# Attempt to import OpenAI; show error if missing
try:
    import openai
except ImportError:
    st.error("The 'openai' package is not installed. Please run 'pip install openai' and restart the app.")
    st.stop()

from collections import defaultdict
from datetime import date
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# OpenAI Config
# --------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your env var is set

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Internal Link Recommender", layout="wide")
st.title("🔗 Internal Link Recommendation Tool")

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
# File Upload
# --------------------------
uploaded_file = st.file_uploader("📄 Upload your CSV file", type="csv")

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.header("⚙️ Settings")
min_similarity_threshold = st.sidebar.slider("Minimum Similarity", 0.0, 1.0, 0.75)
max_outbound_per_source = st.sidebar.number_input("Max Outbound per Source", min_value=1, value=3)

# --------------------------
# Column Mapping
# --------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    col_options = list(df.columns)
    st.sidebar.subheader("📑 Column Mapping")
    url_col = st.sidebar.selectbox("URL column", col_options)
    title_col = st.sidebar.selectbox("Title column", col_options)
    links_col = st.sidebar.selectbox("Links column", col_options)
    embedding_col = st.sidebar.selectbox("Embedding column", col_options)

    df = df.rename(columns={url_col: "URL",
                             title_col: "Title",
                             links_col: "Links",
                             embedding_col: "Embedding"})

    # ----------------------
    # Target Prioritization
    # ----------------------
    st.sidebar.subheader("📌 Target Prioritization")
    target_priority_mode = st.sidebar.selectbox(
        "Prioritize targets by", 
        ["None", "Low inlink count", "URL contains keyword"]
    )
    target_keyword = ""
    if target_priority_mode == "URL contains keyword":
        target_keyword = st.sidebar.text_input("Enter target keyword (e.g. /blog/)")
    target_priority_strength = 1.0
    if target_priority_mode != "None":
        target_priority_strength = st.sidebar.slider(
            "Target Priority Strength", 
            min_value=0.5, max_value=5.0, value=2.0,
            help="Higher values give stronger priority to targeted pages"
        )

    # ----------------------
    # Source Prioritization
    # ----------------------
    st.sidebar.subheader("📌 Source Prioritization")
    source_priority_mode = st.sidebar.selectbox(
        "Prioritize sources by", 
        ["None", "URL contains keyword"]
    )
    source_keyword = ""
    if source_priority_mode == "URL contains keyword":
        source_keyword = st.sidebar.text_input("Enter source keyword (e.g. /section/)")
    source_priority_strength = 1.0
    if source_priority_mode != "None":
        source_priority_strength = st.sidebar.slider(
            "Source Priority Strength", 
            min_value=0.5, max_value=5.0, value=2.0,
            help="Higher values give stronger priority to source pages"
        )

    # --------------------------
    # Main Logic
    # --------------------------
    try:
        # Parse stringified arrays
        df["Links"] = df["Links"].apply(ast.literal_eval)
        df["Embedding"] = df["Embedding"].apply(lambda x: np.array(ast.literal_eval(x)))

        urls = df["URL"].tolist()
        titles = dict(zip(df["URL"], df["Title"]))
        embeddings = np.vstack(df["Embedding"].values)

        # Inlink counts
        inlink_counter = defaultdict(int)
        for source_url, links in zip(df["URL"], df["Links"]):
            for link in links:
                if link.startswith('/'):
                    for url in urls:
                        if url.endswith(link):
                            inlink_counter[url] += 1
                            break
                else:
                    inlink_counter[link] += 1

        if target_priority_mode == "Low inlink count":
            st.subheader("📊 Current Inlink Distribution")
            inlink_df = pd.DataFrame({"URL": list(inlink_counter.keys()),
                                      "Inlink Count": list(inlink_counter.values())})
            st.dataframe(inlink_df.sort_values("Inlink Count"))

        # Existing links set
        existing_links = set()
        for source, links in zip(df["URL"], df["Links"]):
            for target in links:
                if target.startswith('/'):
                    for url in urls:
                        if url.endswith(target):
                            existing_links.add((source, url))
                            break
                else:
                    existing_links.add((source, target))

        # Similarity matrix
        sim_matrix = cosine_similarity(embeddings)

        # Priority boost functions
        def target_priority_boost(target_url):
            if target_priority_mode == "Low inlink count":
                inlink_count = inlink_counter.get(target_url, 0) + 1
                boost = 1 + (target_priority_strength - 1) * (1 / np.log(inlink_count + 1.1))
                return max(1.0, boost)
            elif target_priority_mode == "URL contains keyword" and target_keyword:
                return target_priority_strength if target_keyword.lower() in target_url.lower() else 1.0
            return 1.0

        def source_priority_boost(source_url):
            if source_priority_mode == "URL contains keyword" and source_keyword:
                return source_priority_strength if source_keyword.lower() in source_url.lower() else 1.0
            return 1.0

        outbound_counter = defaultdict(int)
        inbound_counter = defaultdict(int)
        recommendations = []

        st.write(f"Analyzing {len(urls)} pages for linking opportunities...")
        progress_bar = st.progress(0)

        for i, target_url in enumerate(urls):
            progress_bar.progress((i + 1) / len(urls))
            candidate_sims = []
            for j, source_url in enumerate(urls):
                if source_url == target_url or (source_url, target_url) in existing_links:
                    continue
                if outbound_counter[source_url] >= max_outbound_per_source:
                    continue

                sim_score = sim_matrix[j, i]
                if sim_score < min_similarity_threshold:
                    continue

                t_boost = target_priority_boost(target_url)
                s_boost = source_priority_boost(source_url)
                boosted_score = sim_score * t_boost * s_boost

                candidate_sims.append((source_url, target_url, boosted_score, sim_score, t_boost, s_boost))

            candidate_sims.sort(key=lambda x: x[2], reverse=True)
            for source_url, target_url, boosted_score, orig_score, t_boost, s_boost in candidate_sims[:max_outbound_per_source]:
                outbound_counter[source_url] += 1
                inbound_counter[target_url] += 1
                recommendations.append({
                    "SourceURL": source_url,
                    "SourceTitle": titles.get(source_url, ""),
                    "TargetURL": target_url,
                    "TargetTitle": titles.get(target_url, ""),
                    "SimilarityScore": round(orig_score, 4),
                    "TargetBoost": round(t_boost, 2),
                    "SourceBoost": round(s_boost, 2),
                    "BoostedScore": round(boosted_score, 4),
                    "Date": date.today().isoformat(),
                    # will fill anchor text below
                    "AnchorText": ""
                })

        progress_bar.empty()

        # Create DataFrame
        recommendations_df = pd.DataFrame(recommendations)

        # Generate Anchor Text suggestions via OpenAI
        for idx, row in recommendations_df.iterrows():
            prompt = (
                f"Suggest concise anchor text that could fit naturally as part of a sentence but is not a whole sentence for a link from the page titled '{row['SourceTitle']}' "
                f"(URL: {row['SourceURL']}) to the page titled '{row['TargetTitle']}' (URL: {row['TargetURL']})."
            )
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You are an assistant that suggests concise anchor text for hyperlinks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                anchor = resp.choices[0].message.content.strip()
            except Exception:
                anchor = ""
            recommendations_df.at[idx, "AnchorText"] = anchor

        # --------------------------
        # Output
        # --------------------------
        if not recommendations_df.empty:
            if target_priority_mode != "None":
                st.info(f"✨ Target Prioritization: **{target_priority_mode}** (Strength: {target_priority_strength}x)")
            if source_priority_mode != "None":
                st.info(f"🔗 Source Prioritization: **{source_priority_mode}** (Strength: {source_priority_strength}x)")

            st.success(f"✅ Found {len(recommendations_df)} internal link opportunities")
            display_df = recommendations_df.copy()

            cols = ["SourceURL", "SourceTitle", "TargetURL", "TargetTitle", "SimilarityScore"]
            if target_priority_mode != "None": cols += ["TargetBoost"]
            if source_priority_mode != "None": cols += ["SourceBoost"]
            cols += ["BoostedScore", "AnchorText", "Date"]

            display_df = display_df.sort_values("BoostedScore", ascending=False)
            st.dataframe(display_df[cols])

            csv = recommendations_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", data=csv,
                               file_name="recommended_internal_links.csv",
                               mime="text/csv")
        else:
            st.warning("No link recommendations found with current settings.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your input file and settings and try again.")
