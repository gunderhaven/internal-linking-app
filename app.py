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
# 📘 App Description & Help
# --------------------------
st.markdown("""
This tool recommends **internal links** between your site’s pages based on **semantic similarity** using page embeddings.

Upload a single CSV file with all the necessary data. The app will find strong internal linking opportunities between topically related pages.
""")

with st.expander("🛠 How to Generate the Input CSV (using Screaming Frog)", expanded=False):
    st.markdown("""
To use this tool, you’ll need a single CSV with the following columns:

- `URL` — Full URL of the page  
- `Title` — Page title  
- `Links` — Stringified list of internal links from the page (e.g. `"['/about', '/contact']"`)  
- `Embedding` — Stringified list of numbers representing the page’s semantic content (e.g. `"[0.12, 0.33, -0.9, ...]"`)

---

### ✅ Use Screaming Frog to Generate Everything Automatically

Screaming Frog supports **built-in OpenAI integration** to generate embeddings and extract internal links via JavaScript.

---

### 1. **Enable JavaScript Rendering**
Go to `Configuration > Spider > Rendering`, and set it to **JavaScript Rendering**.

---

### 2. **Add a Custom JavaScript Extractor for Internal Links**

Go to `Configuration > Custom > Extraction`, and add a new custom JavaScript extractor:

- **Name:** `Internal Links`
- **Type:** `JavaScript`
- **Script:**

```javascript
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
