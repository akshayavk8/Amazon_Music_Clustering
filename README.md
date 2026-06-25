# 🎵 Amazon Music Clustering

Unsupervised clustering of songs by audio characteristics - grouping \~96,000 tracks into
meaningful mood/genre clusters **without using any genre labels**, using K-Means on
audio features (danceability, energy, acousticness, tempo, etc.), with k selected via
the Elbow Method and Silhouette Score.

🔗 [**Live Demo**](https://amazonmusicclustering-adudeadnd9e2zzftjvfn5x.streamlit.app/) - interactive Streamlit app

\---

## 🔍 Problem

With millions of songs on streaming platforms, manually tagging genres is impractical.
This project automatically groups similar songs based on their audio profile, supporting:

* Personalized playlist curation
* Improved song discovery (similar-track recommendations)
* Artist/catalog analysis
* Market segmentation for streaming platforms

## 📊 Results

|Cluster|Size|Profile|Label|
|-|-|-|-|
|0|13.1%|Very high speechiness (0.83), shortest duration (\~98s)|**Spoken-Word / Short-Form Talk Tracks**|
|1|32.2%|High acousticness (0.75), low energy (0.31)|**Acoustic / Mellow Ballads**|
|2|54.8%|Low acousticness (0.26), high energy (0.69), fastest tempo (125 bpm)|**Upbeat / High-Energy Dance Tracks**|

* **Model:** K-Means, k=3 (selected via Elbow Method + Silhouette Score sweep across k=2–10)
* **Silhouette Score:** ≈ 0.24 · **Davies-Bouldin Index:** ≈ 1.57

Full methodology and charts are in the notebook: [`AmazonMusicClustering.ipynb`](AmazonMusicClustering.ipynb)

## 🗂️ Repository Structure

```
Amazon_Music_Clustering/
|-> AmazonMusicClustering.ipynb             # full pipeline: EDA -> preprocessing -> K-Means -> evaluation -> export
|-> single_genre_artists.csv                # raw input dataset (95,837 songs)
|-> amazon_music_clustered_output.csv       # final output
|-> streamlit_app.py                        # Streamlit app
|-> requirements.txt                        # library dependencies
|-> README.md

```

## 🧰 Tech Stack

Python · Pandas · NumPy · scikit-learn (KMeans, PCA) ·
Matplotlib · Seaborn · Plotly · Streamlit

## 📈 Evaluation Metrics Used

|Metric|Purpose|
|-|-|
|Silhouette Score|How similar a song is to its own cluster vs. others|
|Davies-Bouldin Index|Intra-cluster similarity vs. inter-cluster separation|
|Elbow Method (Inertia)|Selecting the number of clusters k|
|PCA 2D projection|Visual sanity check of cluster separation|

## 🎯 Cluster Interpretation

|Feature|Cluster 0|Cluster 1|Cluster 2|
|-|-|-|-|
|Danceability|0.66|0.49|0.63|
|Energy|0.47|0.31|0.69|
|Acousticness|0.59|0.75|0.26|
|Speechiness|**0.83**|0.06|0.08|
|Valence|0.58|0.41|**0.67**|
|Tempo (bpm)|100|112|**125**|
|Avg. duration|98s|223s|227s|

## 👤 Author

**Akshayaa V. Kumar**
Marine Biologist & Data Science Practitioner
HCL GUVI - Data Science with ML \& AI Certification

[GitHub](https://github.com/akshayavk8) · [LinkedIn](https://linkedin.com/in/akshayavinodkumar)

