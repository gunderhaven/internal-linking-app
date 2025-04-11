import streamlit as st
import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from datetime import date
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Internal Link Recommender", layout="wide")
st.title("🔗 Internal Link Recommendation Tool")

# --------------------------
# Description
# --------------------------
st.markdown("""
This tool recommends **internal links** between your site's pages based on **semantic similarity** using page embeddings.

Upload a single CSV file containing URL data, internal links, and precomputed embeddings.
""")

# --------------------------
# Help Section
# --------------------------
with st.expander("🛠 How to Generate the Input CSV (using Screaming Frog)", expanded=False):
    st.markdown("""
To use this tool, your CSV must contain these columns:

- `URL` - Full URL of the page  
- `Title` - Page title  
- `Links` - Stringified list of internal links from the page (e.g. `"['/about', '/contact']"`)  
- `Embedding` - Stringified list of numbers representing the page's semantic content (e.g. `"[0.12, 0.33, -0.9, ...]"`)

---

### ✅ Use Screaming Frog to Generate Everything Automatically

#### 1. Enable JavaScript Rendering
Go to `Configuration > Spider > Rendering`, and select **JavaScript Rendering**.

---

#### 2. Add a Custom JavaScript Extractor for Internal Links

Go to `Configuration > Custom > Custom Javascript`, and add a new **JavaScript extractor**:

```
var internalLinks = [];

var currentDomain = window.location.hostname;
var allLinks = document.querySelectorAll('a[href]');

allLinks.forEach(function(link) {
    var href = link.getAttribute('href');
    
    if (!href || href.startsWith('javascript:') || href.startsWith('mailto:') || 
        href.startsWith('tel:') || href === '#' || href.startsWith('#')) {
        return;
    }
    
    if (href.startsWith('/') || !href.includes('://')) {
        internalLinks.push(href);
        return;
    }
    
    try {
        var linkDomain = new URL(href).hostname;
        if (linkDomain === currentDomain) {
            internalLinks.push(href);
        }
    } catch (e) {
        // Invalid URL, skip it
    }
});

return seoSpider.data(internalLinks);
```

---

#### 3. Enable OpenAI Embeddings

In the JS extractors library, there are several pre-built extractors that use OpenAI, use this one: (ChatGPT) Extract embeddings from page content. You'll need an OpenAI key, embeddings are dirt-cheap to generate and despite the name this extractor has nothing to do with ChatGPT.

---

#### 4. Run the Crawl and Export

After crawling:

- Export the custom js extractions sheet
- Ensure your CSV includes:
    - `URL`
    - `Title`
    - `Links` (from the JS extractor)
    - `Embedding` (from OpenAI)

You'll probably need to rename your columns before uploading the sheet. This has no tolerance for rewriting columns (work in progress...).
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
MAX_INBOUND = 99999  # disables inbound cap

# Target Prioritization Controls
st.sidebar.subheader("📌 Target Prioritization") 
priority_mode = st.sidebar.selectbox(
    "Prioritize targets by", 
    ["None", "Low inlink count", "URL contains keyword"]
)

keyword = ""
if priority_mode == "URL contains keyword":
    keyword = st.sidebar.text_input("Enter keyword (e.g. /blog/)")

# Priority boost factor - how strong should the bias be
priority_strength = 1.0
if priority_mode != "None":
    priority_strength = st.sidebar.slider(
        "Priority Strength", 
        min_value=0.5, 
        max_value=5.0, 
        value=2.0,
        help="Higher values give stronger priority to targeted pages"
    )

# --------------------------
# Main Logic
# --------------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_columns = ["URL", "Title", "Links", "Embedding"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Error: Missing required columns: {', '.join(missing_columns)}")
            st.stop()
            
        # Parse stringified arrays
        try:
            df["Links"] = df["Links"].apply(ast.literal_eval)
            df["Embedding"] = df["Embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
        except Exception as e:
            st.error(f"Error parsing Links or Embedding columns: {str(e)}")
            st.info("Make sure Links column contains string representations of lists (e.g. '[\"url1\", \"url2\"]') and Embedding column contains string representations of numeric arrays.")
            st.stop()

        urls = df["URL"].tolist()
        titles = dict(zip(df["URL"], df["Title"]))
        embeddings = np.vstack(df["Embedding"].values)

        # Count how many pages link to each page (inlinks)
        inlink_counter = defaultdict(int)
        for source_url, links in zip(df["URL"], df["Links"]):
            for link in links:
                # Some normalization to handle relative/absolute URLs
                if link.startswith('/'):
                    # Try to find a matching URL in our dataset
                    for url in urls:
                        if url.endswith(link):
                            inlink_counter[url] += 1
                            break
                else:
                    inlink_counter[link] += 1

        # Show information about inlink counts if using that prioritization
        if priority_mode == "Low inlink count":
            st.subheader("📊 Current Inlink Distribution")
            inlink_df = pd.DataFrame({
                "URL": list(inlink_counter.keys()),
                "Inlink Count": list(inlink_counter.values())
            })
            inlink_df = inlink_df.sort_values("Inlink Count")
            st.dataframe(inlink_df)

        # Build set of existing links
        existing_links = set()
        for source, links in zip(df["URL"], df["Links"]):
            for target in links:
                # Some normalization to handle relative/absolute URLs
                if target.startswith('/'):
                    # Try to find a matching URL in our dataset
                    for url in urls:
                        if url.endswith(target):
                            existing_links.add((source, url))
                            break
                else:
                    existing_links.add((source, target))

        # Compute pairwise similarity
        sim_matrix = cosine_similarity(embeddings)

        # Target priority boost function
        def target_priority_boost(target_url):
            if priority_mode == "Low inlink count":
                # Lower inlink count = higher weight
                # Add 1 to prevent division by zero
                inlink_count = inlink_counter.get(target_url, 0) + 1
                # Normalize the weight using logarithmic scaling for better distribution
                boost = 1 + (priority_strength - 1) * (1 / np.log(inlink_count + 1.1))
                return max(1.0, boost)  # Ensure minimum boost is 1.0
                
            elif priority_mode == "URL contains keyword" and keyword:
                # If URL contains keyword, apply the priority strength as boost
                return priority_strength if keyword.lower() in target_url.lower() else 1.0
            else:
                return 1.0  # No boost

        outbound_counter = defaultdict(int)
        inbound_counter = defaultdict(int)
        recommendations = []

        # Debug information
        st.write(f"Analyzing {len(urls)} pages for linking opportunities...")
        progress_bar = st.progress(0)

        for i, target_url in enumerate(urls):
            # Update progress
            progress_bar.progress((i + 1) / len(urls))
            
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

                # Apply priority boost
                boost = target_priority_boost(target_url)
                boosted_score = sim_score * boost
                
                # Store both the original similarity and boosted score
                candidate_sims.append((source_url, target_url, boosted_score, sim_score, boost))

            # Sort by boosted score but keep track of original similarity
            candidate_sims.sort(key=lambda x: x[2], reverse=True)

            for source_url, target_url, boosted_score, orig_score, boost in candidate_sims[:max_outbound_per_source]:
                outbound_counter[source_url] += 1
                inbound_counter[target_url] += 1
                recommendations.append({
                    "SourceURL": source_url,
                    "SourceTitle": titles.get(source_url, ""),
                    "TargetURL": target_url,
                    "TargetTitle": titles.get(target_url, ""),
                    "SimilarityScore": round(orig_score, 4),
                    "PriorityBoost": round(boost, 2),
                    "BoostedScore": round(boosted_score, 4),
                    "Status": "",
                    "Date": date.today().isoformat(),
                    "Notes": ""
                })

        progress_bar.empty()
        
        # --------------------------
        # Output
        # --------------------------
        if recommendations:
            recommendations_df = pd.DataFrame(recommendations)
            
            # Show information about prioritization if active
            if priority_mode != "None":
                st.info(f"✨ Target Prioritization: **{priority_mode}** (Boost Strength: {priority_strength}x)")
                if priority_mode == "URL contains keyword" and keyword:
                    st.info(f"🔍 Prioritizing targets containing: **{keyword}**")
            
            st.success(f"✅ Found {len(recommendations_df)} internal link opportunities")
            
            # Create display columns that better explain the score boosting
            display_df = recommendations_df.copy()
            
            # Determine columns to display based on prioritization
            if priority_mode != "None":
                columns_to_display = [
                    "SourceURL", "SourceTitle", "TargetURL", "TargetTitle", 
                    "SimilarityScore", "PriorityBoost", "BoostedScore",
                    "Status", "Date"
                ]
                
                # Sort by boosted score for clearer prioritization
                display_df = display_df.sort_values("BoostedScore", ascending=False)
            else:
                # Hide boost-related columns if not using prioritization
                columns_to_display = [
                    "SourceURL", "SourceTitle", "TargetURL", "TargetTitle", 
                    "SimilarityScore", "Status", "Date"
                ]
                
                # Sort by similarity score when no prioritization
                display_df = display_df.sort_values("SimilarityScore", ascending=False)
                
            st.dataframe(display_df[columns_to_display])

            # Include all columns in the download for analysis
            csv = recommendations_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", data=csv, file_name="recommended_internal_links.csv", mime="text/csv")
        else:
            st.warning("No link recommendations found with current settings. Try adjusting the similarity threshold or prioritization settings.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your input file format and try again.")
