import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from pathlib import Path

st.set_page_config(page_title="Mapa de Clusters • CDs", layout="wide")

DATA_DIR = Path("DataBase")
POINTS_FILE = DATA_DIR / "points_enriched_final.parquet"
CLUSTERS_FILE = DATA_DIR / "clusters_summary_final.parquet"

# -------- Utils
@st.cache_data(show_spinner=False)
def load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        # tenta nomes alternativos comuns
        alts = list(DATA_DIR.glob("points_enriched*.parquet")) + list(DATA_DIR.glob("clusters_summary*.parquet"))
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            f"Arquivos disponíveis em {DATA_DIR.resolve()}:\n - " + "\n - ".join([p.name for p in alts])
        )
    return pd.read_parquet(path)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# -------- Carregamento
points = load_parquet_safe(POINTS_FILE)
clusters = load_parquet_safe(CLUSTERS_FILE)

# sanity/renome
points.columns = [c.strip().lower() for c in points.columns]
clusters.columns = [c.strip().lower() for c in clusters.columns]

# garante colunas
base_points_cols = {'cluster','latitude','longitude'}
base_clusters_cols = {'cluster','centroid_lat','centroid_lon','n_points'}
assert base_points_cols.issubset(points.columns), f"Faltam colunas em points: {base_points_cols - set(points.columns)}"
assert base_clusters_cols.issubset(clusters.columns), f"Faltam colunas em clusters: {base_clusters_cols - set(clusters.columns)}"

# radius_km pode não existir (calcula on-the-fly)
if 'radius_km' not in clusters.columns:
    tmp = points.merge(clusters[['cluster','centroid_lat','centroid_lon']], on='cluster', how='left', validate='m:1')
    tmp['dist_km'] = haversine_km(tmp['latitude'], tmp['longitude'], tmp['centroid_lat'], tmp['centroid_lon'])
    clusters = clusters.merge(
        tmp.groupby('cluster')['dist_km'].quantile(0.9).rename('radius_km').reset_index(),
        on='cluster', how='left'
    )

# region_macro/cd_name opcionais
if 'region_macro' not in clusters.columns:
    clusters['region_macro'] = 'Indefinido'
if 'cd_name' not in clusters.columns:
    clusters['cd_name'] = clusters['cluster'].apply(lambda c: f"CD – Cluster {int(c):02d}")

clusters['radius_m'] = (clusters['radius_km'].fillna(0)*1000).astype(float)

# -------- Sidebar / Filtros
st.sidebar.header("Filtros")

cluster_opts = sorted(clusters['cluster'].astype(int).unique().tolist())
sel_clusters = st.sidebar.multiselect("Clusters", cluster_opts, default=cluster_opts)

region_opts = sorted(clusters['region_macro'].dropna().unique().tolist())
sel_regions = st.sidebar.multiselect("Macro-região", region_opts, default=region_opts)

max_points = int(st.sidebar.number_input("Máx. de pontos no mapa (amostra)", min_value=500, max_value=50000, value=8000, step=500))
show_points = st.sidebar.checkbox("Mostrar pontos", value=True)
show_areas  = st.sidebar.checkbox("Mostrar áreas (raio p90)", value=True)
show_centers = st.sidebar.checkbox("Mostrar centróides", value=True)

points_f = points.copy()
clusters_f = clusters.copy()

points_f = points_f[points_f['cluster'].astype(int).isin(sel_clusters)]
clusters_f = clusters_f[clusters_f['cluster'].astype(int).isin(sel_clusters)]

if sel_regions:
    clusters_f = clusters_f[clusters_f['region_macro'].isin(sel_regions)]
    points_f = points_f[points_f['cluster'].isin(clusters_f['cluster'])]

# -------- KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clusters", int(clusters_f['cluster'].nunique()))
col2.metric("Pontos", int(points_f.shape[0]))
col3.metric("Raio p90 médio (km)", f"{clusters_f['radius_km'].mean():.2f}")
col4.metric("Maior raio p90 (km)", f"{clusters_f['radius_km'].max():.2f}")

# -------- Centro do mapa
center_lat = float(clusters_f['centroid_lat'].mean())
center_lon = float(clusters_f['centroid_lon'].mean())

layers = []

if show_points:
    pts = points_f
    if len(pts) > max_points:
        pts = pts.sample(max_points, random_state=42)
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position='[longitude, latitude]',
            get_radius=50,            # metros
            radius_units="meters",
            pickable=True,
            get_fill_color=[30, 144, 255, 120],   # azul translúcido
        )
    )

if show_areas:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=clusters_f.assign(radius_m=clusters_f['radius_m'].clip(lower=200)),
            get_position='[centroid_lon, centroid_lat]',
            get_radius="radius_m",
            radius_units="meters",
            pickable=True,
            get_fill_color=[255, 99, 71, 60],     # laranja translúcido
            get_line_color=[255, 99, 71, 160],
            line_width_min_pixels=1,
        )
    )

if show_centers:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=clusters_f,
            get_position='[centroid_lon, centroid_lat]',
            get_radius=70,
            radius_units="meters",
            get_fill_color=[220, 20, 60, 200],    # vermelho
            pickable=True,
        )
    )

tooltip = {
    "html": (
        "<b>{cd_name}</b><br/>"
        "Cluster: {cluster}<br/>"
        "Macro-região: {region_macro}<br/>"
        "Raio p90: {radius_km} km<br/>"
        "Pontos: {n_points}"
    ),
    "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"}
}

mapbox_key = st.secrets.get("MAPBOX_API_KEY", "")
deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    mapbox_key=mapbox_key,
    initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=5),
    layers=layers,
    tooltip=tooltip,
)

st.subheader("Mapa de Clusters e Áreas de Cobertura (p90)")
st.pydeck_chart(deck, use_container_width=True)

# -------- Tabelas
st.markdown("### Clusters selecionados")
st.dataframe(
    clusters_f[['cluster','cd_name','region_macro','radius_km','n_points']].sort_values('cluster'),
    use_container_width=True
)

# download filtrado
csv = clusters_f.to_csv(index=False).encode("utf-8")
st.download_button("Baixar resumo de clusters (CSV)", data=csv, file_name="clusters_summary_filtrado.csv", mime="text/csv")
