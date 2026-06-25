"""
Amazon Music Clustering — Streamlit Explorer
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Search common locations so this works whether streamlit_app.py lives at the
# repo root or inside an app/ subfolder.
_CANDIDATE_PATHS = [
    os.path.join(APP_DIR, "data", "amazon_music_clustered_output.csv"),       # script/data/
    os.path.join(APP_DIR, "..", "data", "amazon_music_clustered_output.csv"), # script/../data/
    os.path.join(APP_DIR, "amazon_music_clustered_output.csv"),               # same folder as script
]
DATA_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[0])

st.set_page_config(page_title="Amazon Music Clustering", layout="wide", page_icon="🎵")

FEATURES = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "duration_ms"]

CLUSTER_LABELS = {
    0: "Spoken-Word / Short-Form Talk Tracks",
    1: "Acoustic / Mellow Ballads",
    2: "Upbeat / High-Energy Dance Tracks",
}


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Couldn't find the dataset. Checked these locations:\n\n"
            + "\n".join(f"- `{p}`" for p in _CANDIDATE_PATHS)
            + "\n\nMake sure `amazon_music_clustered_output.csv` is committed to your "
              "GitHub repo (check it isn't excluded by `.gitignore`, and that the file "
              "actually appears on github.com in your repo, not just locally)."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def get_pca(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_


df = load_data()
df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS).fillna(df["cluster"].astype(str))

st.title("🎵 Amazon Music Clustering Explorer")
st.markdown(
    "Unsupervised K-Means clustering of songs by **audio characteristics** "
    "(danceability, energy, acousticness, tempo, etc.) — no genre labels used as input."
)

# ---- Sidebar filters ----
st.sidebar.header("Filters")
selected_clusters = st.sidebar.multiselect(
    "Cluster(s)", options=sorted(df["cluster"].unique()),
    default=sorted(df["cluster"].unique()),
    format_func=lambda c: f"Cluster {c} — {CLUSTER_LABELS.get(c, '')}"
)
sample_size = st.sidebar.slider("Plot sample size", 1000, 15000, 6000, step=1000)
search_artist = st.sidebar.text_input("Search artist name")

df_filtered = df[df["cluster"].isin(selected_clusters)]
if search_artist:
    df_filtered = df_filtered[df_filtered["name_artists"].str.contains(search_artist, case=False, na=False)]

# ---- KPI row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Songs", f"{len(df):,}")
col2.metric("Clusters", df["cluster"].nunique())
col3.metric("Filtered Songs", f"{len(df_filtered):,}")
col4.metric("Artists", df_filtered["name_artists"].nunique())

st.divider()

# ---- Tabs ----
tab1, tab2, tab3, tab4 = st.tabs(["📊 Cluster Overview", "🗺️ PCA Visualization", "🔍 Explore Songs", "🎯 Find Similar Songs"])

with tab1:
    st.subheader("Cluster Sizes")
    size_df = df["cluster"].value_counts().sort_index().reset_index()
    size_df.columns = ["cluster", "count"]
    size_df["label"] = size_df["cluster"].map(CLUSTER_LABELS)
    fig_bar = px.bar(size_df, x="label", y="count", color="label", text="count",
                      labels={"label": "Cluster", "count": "Number of Songs"})
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Average Feature Profile by Cluster")
    profile = df.groupby("cluster_label")[FEATURES].mean().round(3)
    st.dataframe(profile, use_container_width=True)

    norm_feats = ["danceability", "energy", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence"]
    fig_radar = px.line_polar(
        profile[norm_feats].reset_index().melt(id_vars="cluster_label", var_name="feature", value_name="value"),
        r="value", theta="feature", color="cluster_label", line_close=True,
        title="Feature Profile Comparison (Radar)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.subheader("Songs in PCA Space")
    coords, var_ratio = get_pca(df)
    df_plot = df.copy()
    df_plot["pca1"] = coords[:, 0]
    df_plot["pca2"] = coords[:, 1]
    plot_sample = df_plot[df_plot["cluster"].isin(selected_clusters)].sample(
        min(sample_size, len(df_plot[df_plot["cluster"].isin(selected_clusters)])), random_state=1
    )
    fig_scatter = px.scatter(
        plot_sample, x="pca1", y="pca2", color="cluster_label",
        hover_data=["name_song", "name_artists"],
        labels={"pca1": f"PC1 ({var_ratio[0]*100:.1f}% var)", "pca2": f"PC2 ({var_ratio[1]*100:.1f}% var)"},
        title="K-Means Clusters Visualized via PCA"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Browse Songs")
    sort_col = st.selectbox("Sort by", ["popularity_songs", "danceability", "energy", "tempo", "valence"])
    show_cols = ["name_song", "name_artists", "cluster_label"] + FEATURES
    st.dataframe(
        df_filtered[show_cols].sort_values(sort_col, ascending=False).head(200),
        use_container_width=True, height=500
    )
    st.download_button(
        "Download filtered results as CSV",
        df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_clusters.csv", mime="text/csv"
    )

with tab4:
    st.subheader("Find Songs Similar to a Track")
    track_choice = st.selectbox("Pick a song", df["name_song"].dropna().unique()[:5000])
    if track_choice:
        ref_row = df[df["name_song"] == track_choice].iloc[0]
        ref_cluster = ref_row["cluster"]
        st.write(f"**{track_choice}** by {ref_row['name_artists']} — Cluster: *{CLUSTER_LABELS.get(ref_cluster)}*")

        scaler = StandardScaler()
        X_all = scaler.fit_transform(df[FEATURES])
        ref_idx = df[df["name_song"] == track_choice].index[0]
        dists = np.linalg.norm(X_all - X_all[ref_idx], axis=1)
        df_sim = df.copy()
        df_sim["distance"] = dists
        similar = df_sim[df_sim.index != ref_idx].sort_values("distance").head(10)
        st.dataframe(similar[["name_song", "name_artists", "cluster_label", "distance"] + FEATURES],
                     use_container_width=True)

st.divider()
