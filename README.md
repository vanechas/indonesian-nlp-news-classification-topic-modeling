# Indonesian NLP Pipeline: News Classification & Opinion Mining

An end-to-end Natural Language Processing (NLP) repository implementing asynchronous web scraping, text preprocessing, deep learning classification with IndoBERT, and unsupervised opinion clustering with topic modeling on Indonesian text.

---

## 📌 Project Overview

This repository addresses two primary natural language tasks:

### 1. Sports News Scraping & Hierarchical Classification
* **Data Scraping:** High-throughput asynchronous crawler (`httpx`, `asyncio`, `BeautifulSoup`) collecting Indonesian sports news from **Detik.com**, **Antara News**, and **Liputan6** across 5 categories: *Non-football sports*, *Liga Indonesia*, *English Premier League*, *La Liga*, and *Serie A*.
* **Preprocessing:** Noise filtering, stopword removal, and vectorized regex-based Indonesian pseudo-stemming.
* **Classification Models:** Fine-tuned **IndoBERT** benchmarked under two experimental setups:
  * **1-Stage Architecture:** Direct 5-class classification.
  * **2-Stage Hierarchical Architecture:** Stage 1 (Football vs. Non-football) followed by Stage 2 (League-level classification).
* **Optimization:** Hyperparameter tuning evaluating learning rates (`2e-5` vs. `5e-5`) and batch sizes (`16` vs. `32`).

### 2. YouTube Comment Mining & Persona Topic Extraction
* **Data Extraction:** Automated retrieval of Indonesian YouTube discussions on the impact of AI in education using `youtube-comment-downloader`.
* **Text Representation:** Feature extraction comparing **TF-IDF** and **Bag of Words (CountVectorizer)**.
* **Unsupervised Clustering:** **K-Means Clustering** evaluated across cluster counts ($k \in [2, 5]$) via **Silhouette Score analysis** to discover user personas.
* **Topic Modeling:** **Latent Dirichlet Allocation (LDA)** to extract latent semantic structures and public sentiment regarding AI in academic workflows and career displacement.

---

## 📊 Key Results

### Classification (IndoBERT)
* **Best Architecture:** 1-Stage Model (LR: `2e-5`, Batch Size: `16`) achieved **97.97% accuracy** (Macro F1: **0.98**).
* **Comparison:** The 1-stage model marginally outperformed the 2-stage hierarchical model (97.30% accuracy) due to cascading error propagation in multi-stage inference.

### Clustering & Topic Modeling (AI in Education)
* **Optimal Representation:** Bag-of-Words with **$k=2$** produced the highest Silhouette Score (**0.6189**), indicating robust cluster separation.
* **Identified Personas / Topics:**
  * **Persona 1 (Broad Impact & Existential Critique):** Public discourse surrounding ethical limits, control, and fear of human replacement by AI.
  * **Persona 2 (Educational & Labor Utility):** Perspectives from educators and students focusing on classroom efficiency, automated assignments, and workforce transition.

---

## 🛠️ Tech Stack & Tools

* **Languages:** Python (Jupyter Notebook)
* **Web Scraping & Async:** `httpx`, `asyncio`, `BeautifulSoup4`, `youtube-comment-downloader`
* **NLP & Feature Engineering:** `Sastrawi`, `scikit-learn` (`TfidfVectorizer`, `CountVectorizer`)
* **Machine Learning & Deep Learning:** `IndoBERT` (Transformers), `KMeans`, `LatentDirichletAllocation`
* **Evaluation & Visualization:** `matplotlib`, `seaborn`, `scikit-learn` (`silhouette_score`, `classification_report`)
