import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import date

st.title("🔗 Internal Link Recommendation Tool")

st.markdown("Upload your datasets below:")

# Uploads
pages_file = st.file_uploader("1. Upload Pages CSV (must include 'URL' & 'Title' columns)", type="csv")
similarity_file = st.file_uploader("2. Upload Similarity Matrix (.npy)", type="npy")
existing_links_file = st.file_uploader("3. Upload Existing Links CSV (columns: SourceURL, TargetURL)", type="csv")
inlink_counts_file = st.file_uploader("4. Upload Inlink Counts CSV (columns: URL, InlinkCount)", type="csv")

# Parameter controls
st.sidebar.header("🔧 Settings")
min_similarity_threshold = st.sidebar.slider("Minimum Similarity", 0.0, 1.0, 0.75)
max_outbound = st.sidebar.number_input("Max Outbound per Source", min_value=1, value=3)
max_inbound = st.sidebar.number_input("Max Inbound per Target", min_value=1, value=5)

if all([pages_file, similarity_file, existing_links_file, inlink_counts_file]):
    st.success("✅ All files uploaded. Ready to generate recommendations!")

    pages_df = pd.read_csv(pages_file)
    pages_df.set_index("URL", inplace=True)

    similarity_matrix = np.load(similarity_file)
    urls = list(pages_df.index)

    existing_links_df = pd.read_csv(existing_links_file)
    existing_links = set(tuple(x) for x in existing_links_df.to_records(index=False))

    inlink_counts = pd.read_csv(inlink_counts_file).set_index("URL")["InlinkCount"].to_dict()

    if st.button("🚀 Generate Link Recommendations"):
        outbound_counter = defaultdict(int)
        inbound_counter = defaultdict(int)
        recommendations = []

        for i, target_url in enumerate(urls):
            if inlink_counts.get(target_url, 0) >= max_inbound:
                continue

            candidate_sims = []
            for j, source_url in enumerate(urls):
                if source_url == target_url:
                    continue
                if (source_url, target_url) in existing_links:
                    continue
                if outbound_counter[source_url] >= max_outbound:
                    continue
                sim_score = similarity_matrix[j, i]
                if sim_score < min_similarity_threshold:
                    continue
                candidate_sims.append((source_url, target_url, sim_score))

            candidate_sims.sort(key=lambda x: x[2], reverse=True)
            for source_url, target_url, score in candidate_sims[:max_outbound]:
                outbound_counter[source_url] += 1
                inbound_counter[target_url] += 1
                recommendations.append({
                    "SourceURL": source_url,
                    "SourceTitle": pages_df.loc[source_url, "Title"],
                    "TargetURL": target_url,
                    "TargetTitle": pages_df.loc[target_url, "Title"],
                    "SimilarityScore": score,
                    "Status": "",
                    "Date": date.today().isoformat(),
                    "Notes": ""
                })

        recommendations_df = pd.DataFrame(recommendations)
        st.success(f"✅ {len(recommendations_df)} internal link opportunities found")
        st.dataframe(recommendations_df)

        csv = recommendations_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="recommended_internal_links.csv", mime="text/csv")
