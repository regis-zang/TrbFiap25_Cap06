# app.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---- (opcional) diagnose de versões
st.caption(f"Python: {sys.version.split()[0]}")
try:
    import numpy as _np, pandas as _pd, fastparquet as _fp  # noqa: F401
    st.caption(f"NumPy: {_np.__version__} | Pandas: {_pd.__version__} | Parquet: fastparquet ✅")
except Exception:
    st.caption("Parquet engine: (não detectado) ⚠️")

# =========================
# Config
# =========================
st.set_page_config(page_title="Mapa de Clusters • CDs", layout="wide")
DATA_DIR = Path("DataBase")
POINTS_FILE = DATA_DIR / "points_enriched_final.parquet"
CLUSTERS_FILE = DATA_DIR / "clusters_summary_final.parquet"

# =========================
# Utils
# =========================
@st.cache_data(show_spinner=False)
def load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        existing = [p.name for p in DATA_DIR.glob("*.parquet")]
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            f"Disponíveis em {DATA_DIR.resolve()}:\n - " + "\n - ".join(existing)
        )
    # tenta engine default (deve pegar fastparquet)
    try:
        return pd.read_parquet(path)
    except Exception:
        # força fastparquet explicitamente
        return pd.read_parquet(path, engine="fastparquet")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def color_by_region(region: str) -> list:
    pal = {
        "Norte":        [ 26, 188, 156, 180],
        "Nordeste":     [241, 196,  15, 180],
        "Centro-Oeste": [142,  68, 173, 180],
        "Sudeste":      [ 52, 152, 219, 180],
        "Sul":          [231,  76,  60, 180],
        "Indefinido":   [127, 140, 141, 180],
    }
    return pal.get(region, pal["Indefinido"])

def compute_view_safe(lons, lats):
    """Tenta usar compute_view do pydeck; se não existir, faz um fallback simples."""
    try:
        from pydeck.data_utils import compute_view
        df = pd.DataFrame({"lon": lons, "lat": lats})
        view = compute_view(df[["lon", "lat"]])
        try:
            view.zoom = min(9, max(3, float(view.zoom) + 0.5))
        except Exception:
            view.zoom = 6
        return view
    except Exception:
        return pdk.ViewState(latitude=float(np.mean(lats)),
                             longitude=float(np.mean(lons)),
                             zoom=6)

def build_deck_resilient(layers, view_state, tooltip_cfg):
    """Cria o pdk.Deck com Mapbox (se houver token), senão Carto, senão sem base map."""
    deck_kwargs = dict(initial_view_state=view_state, layers=layers)
    # tenta incluir tooltip; se a versão não suportar, remove
    try:
        deck_kwargs["tooltip"] = tooltip_cfg
    except Exception:
        pass

    # token via secrets (se houver)
    mapbox_key = st.secrets.get("MAPBOX_API_KEY", "").strip()
    if mapbox_key:
        try:
            pdk.settings.mapbox_api_key = mapbox_key
        except Exception:
            pass

    try:
        return pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", **deck_kwargs)
    except TypeError:
        try:
            return pdk.Deck(map_provider="carto", map_style="light", **deck_kwargs)
        except TypeError:
            deck_kwargs.pop("tooltip", None)
            return pdk.Deck(**deck_kwargs)

# =========================
# Load
# =========================
points = load_parquet_safe(POINTS_FILE)
clusters = load_parquet_safe(CLUSTERS_FILE)

# normaliza nomes
points.columns = [c.strip().lower() for c in points.columns]
clusters.columns = [c.strip().lower() for c in clusters.columns]

# sanity
req_points = {'cluster','latitude','longitude'}
req_clusters = {'cluster','centroid_lat','centroid_lon','n_points'}
missing_p = req_points - set(points.columns)
missing_c = req_clusters - set(clusters.columns)
if missing_p:
    st.error(f"Faltam colunas em points: {missing_p}")
    st.stop()
if missing_c:
    st.error(f"Faltam colunas em clusters: {missing_c}")
    st.stop()

# radius_km (p90) se não existir
if 'radius_km' not in clusters.columns:
    tmp = points.merge(
        clusters[['cluster','centroid_lat','centroid_lon']],
        on='cluster', how='left', validate='m:1'
    )
    tmp['dist_km'] = haversine_km(tmp['latitude'], tmp['longitude'],
                                  tmp['centroid_lat'], tmp['centroid_lon'])
    clusters = clusters.merge(
        tmp.groupby('cluster')['dist_km'].quantile(0.9).rename('radius_km').reset_index(),
        on='cluster', how='left'
    )

# region_macro & cd_name
if 'region_macro' not in clusters.columns:
    clusters['region_macro'] = 'Indefinido'
if 'cd_name' not in clusters.columns:
    clusters['cd_name'] = clusters['cluster'].apply(lambda c: f"CD – Cluster {int(c):02d}")

clusters['radius_m'] = (clusters['radius_km'].fillna(0)*1000).astype(float)

# metadata para points
points = points.merge(clusters[['cluster','cd_name','region_macro']],
                      on='cluster', how='left')

# =========================
# Sidebar / filtros
# =========================
st.sidebar.header("Filtros")

cluster_opts = sorted(clusters['cluster'].astype(int).unique().tolist())
sel_clusters = st.sidebar.multiselect("Clusters", cluster_opts, default=cluster_opts)

region_opts = sorted(clusters['region_macro'].dropna().unique().tolist())
sel_regions = st.sidebar.multiselect("Macro-região", region_opts, default=region_opts)

max_points = int(st.sidebar.number_input("Máx. de pontos no mapa (amostra)",
                                         min_value=500, max_value=50000, value=7000, step=500))
show_points    = st.sidebar.checkbox("Mostrar pontos", value=True)
show_areas     = st.sidebar.checkbox("Mostrar áreas (raio p90)", value=True)
show_centroids = st.sidebar.checkbox("Mostrar centróides", value=True)
color_points_by_region = st.sidebar.checkbox("Colorir pontos por macro-região", value=False)

# =========================
# Filtragem
# =========================
points_f = points[points['cluster'].astype(int).isin(sel_clusters)].copy()
clusters_f = clusters[clusters['cluster'].astype(int).isin(sel_clusters)].copy()
if sel_regions:
    clusters_f = clusters_f[clusters_f['region_macro'].isin(sel_regions)]
    points_f = points_f[points_f['cluster'].isin(clusters_f['cluster'])]

# =========================
# KPIs
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clusters", int(clusters_f['cluster'].nunique()))
c2.metric("Pontos", int(points_f.shape[0]))
c3.metric("Raio p90 médio (km)", f"{clusters_f['radius_km'].mean():.2f}")
c4.metric("Maior raio p90 (km)", f"{clusters_f['radius_km'].max():.2f}")

# =========================
# Mapa
# =========================
st.subheader("Mapa de Clusters e Áreas de Cobertura (p90)")

layers = []
if not clusters_f.empty:
    # -------- view automático
    lons = pd.concat(
        [
            points_f['longitude'],
            clusters_f['centroid_lon']
        ],
        ignore_index=True
    )
    lats = pd.concat(
        [
            points_f['latitude'],
            clusters_f['centroid_lat']
        ],
        ignore_index=True
    )
    view = compute_view_safe(lons, lats)

    # -------- pontos (amostra)
    if show_points and not points_f.empty:
        pts = points_f
        if len(pts) > max_points:
            pts = pts.sample(max_points, random_state=42)

        if color_points_by_region:
            pts = pts.copy()
            pts['rgb'] = pts['region_macro'].apply(color_by_region)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pts,
                    get_position='[longitude, latitude]',
                    get_radius=120,  # maior pra aparecer melhor no zoom inicial
                    radius_units="meters",
                    pickable=True,
                    get_fill_color='rgb',
                )
            )
        else:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pts,
                    get_position='[longitude, latitude]',
                    get_radius=120,
                    radius_units="meters",
                    pickable=True,
                    get_fill_color=[30, 144, 255, 120],
                )
            )

    # -------- áreas (raio p90)
    if show_areas:
        areas = clusters_f.assign(radius_m=clusters_f['radius_m'].clip(lower=200)).copy()
        areas['rgb'] = areas['region_macro'].apply(color_by_region)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=areas,
                get_position='[centroid_lon, centroid_lat]',
                get_radius="radius_m",
                radius_units="meters",
                pickable=True,
                get_fill_color='rgb',
                get_line_color=[50, 50, 50, 200],
                line_width_min_pixels=1,
            )
        )

    # -------- centróides
    if show_centroids:
        centers = clusters_f.copy()
        centers['rgb'] = centers['region_macro'].apply(color_by_region)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=centers,
                get_position='[centroid_lon, centroid_lat]',
                get_radius=90,
                radius_units="meters",
                get_fill_color='rgb',
                pickable=True,
            )
        )

    tooltip_cfg = {
        "html": (
            "<b>{cd_name}</b><br/>"
            "Cluster: {cluster}<br/>"
            "Macro-região: {region_macro}<br/>"
            "Raio p90: {radius_km} km<br/>"
            "Pontos: {n_points}"
        ),
        "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"},
    }

    deck = build_deck_resilient(layers, view, tooltip_cfg)
    st.caption("Base map: " + ("Mapbox ✅" if bool(st.secrets.get('MAPBOX_API_KEY', '').strip())
                               else "Carto/No base ✅"))
    st.pydeck_chart(deck, use_container_width=True)
else:
    st.warning("Nenhum cluster selecionado para exibir no mapa.")

# =========================
# Tabela + download
# =========================
st.markdown("### Clusters selecionados")
st.dataframe(
    clusters_f[['cluster','cd_name','region_macro','radius_km','n_points']].sort_values('cluster'),
    use_container_width=True
)

csv = clusters_f.to_csv(index=False).encode("utf-8")
st.download_button("Baixar resumo de clusters (CSV)",
                   data=csv, file_name="clusters_summary_filtrado.csv", mime="text/csv")
