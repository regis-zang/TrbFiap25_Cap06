# app.py
import sys, json, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ======= PRIMEIRA CHAMADA DO STREAMLIT =======
st.set_page_config(page_title="Mapa de Clusters • CDs", layout="wide")
# =============================================

# (diagnóstico curto)
st.caption(f"Python: {sys.version.split()[0]}")

# ---------------- Config ----------------
DATA_DIR = Path("DataBase")
POINTS_FILE   = DATA_DIR / "points_enriched_final.parquet"
CLUSTERS_FILE = DATA_DIR / "clusters_summary_final.parquet"
UF_GEOJSON    = DATA_DIR / "br_estados.geojson"       # limites de UF (GeoJSON)

UF_FALLBACK_URL = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
)

# ---------------- Utils ----------------
@st.cache_data(show_spinner=False)
def load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        existing = [p.name for p in DATA_DIR.glob("*.parquet")]
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            f"Disponíveis em {DATA_DIR.resolve()}:\n - " + "\n - ".join(existing)
        )
    try:
        return pd.read_parquet(path)    # fastparquet por default
    except Exception:
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

def clean_geo(df: pd.DataFrame, lon: str, lat: str) -> pd.DataFrame:
    out = df.copy()
    out[lon] = pd.to_numeric(out[lon], errors="coerce")
    out[lat] = pd.to_numeric(out[lat], errors="coerce")
    out = out.dropna(subset=[lon, lat])
    out = out[out[lon].between(-180, 180) & out[lat].between(-90, 90)]
    return out

def compute_view_safe(lons, lats):
    try:
        from pydeck.data_utils import compute_view
        df = pd.DataFrame({"lon": lons, "lat": lats})
        if df.empty:
            return pdk.ViewState(latitude=-14.2, longitude=-51.9, zoom=3.5)
        view = compute_view(df[["lon", "lat"]])
        try:
            view.zoom = min(9, max(3, float(view.zoom) + 0.8))
        except Exception:
            view.zoom = 6
        return view
    except Exception:
        if len(lats) == 0:
            return pdk.ViewState(latitude=-14.2, longitude=-51.9, zoom=3.5)
        return pdk.ViewState(latitude=float(np.mean(lats)),
                             longitude=float(np.mean(lons)),
                             zoom=6)

def build_deck_resilient(layers, view_state, tooltip_cfg):
    deck_kwargs = dict(initial_view_state=view_state, layers=layers)
    deck_kwargs["tooltip"] = tooltip_cfg

    # Mapbox opcional
    mapbox_key = st.secrets.get("MAPBOX_API_KEY", "").strip()
    if mapbox_key:
        try:
            pdk.settings.mapbox_api_key = mapbox_key
        except Exception:
            pass

    # tenta Mapbox; cai para Carto; se falhar, sem basemap
    try:
        return pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", **deck_kwargs)
    except Exception:
        try:
            return pdk.Deck(map_provider="carto", map_style="light", **deck_kwargs)
        except Exception:
            deck_kwargs.pop("tooltip", None)
            return pdk.Deck(**deck_kwargs)

# ---------- GeoPandas (opcional) + helpers ----------
try:
    import geopandas as gpd
    GEOPANDAS_OK = True
except Exception:
    GEOPANDAS_OK = False

def load_uf_geojson_gdf(path: Path):
    """Carrega GeoJSON sem Fiona (via json) e monta um GeoDataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    # coluna UF
    for cand in ["sigla", "UF", "uf", "sigla_uf", "SIGLA_UF", "name", "NM_UF", "NOME_UF", "state_code"]:
        if cand in gdf.columns:
            gdf["uf_col"] = gdf[cand].astype(str)
            break
    else:
        gdf["uf_col"] = gdf.iloc[:, 0].astype(str)
    return gdf

def lerp_color(a, b, t):
    """Interpolação linear RGBA entre duas cores."""
    return [int(a[i] + (b[i]-a[i])*t) for i in range(4)]

def choropleth_color(val, vmin, vmax):
    if pd.isna(val):
        return [220, 220, 220, 60]
    if vmax == vmin:
        t = 1.0
    else:
        t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    start = [255, 255, 178, 90]   # amarelo claro
    end   = [189,   0,  38, 180]  # vermelho escuro
    return lerp_color(start, end, t)

def ensure_uf_geojson():
    """Baixa o GeoJSON de UF se não existir (sem dependências extras)."""
    if UF_GEOJSON.exists():
        return True, "já existe"
    try:
        UF_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UF_FALLBACK_URL, timeout=30) as r:
            data = r.read()
        UF_GEOJSON.write_bytes(data)
        return True, "baixado"
    except Exception as e:
        return False, str(e)

# ---------------- Load ----------------
points   = load_parquet_safe(POINTS_FILE)
clusters = load_parquet_safe(CLUSTERS_FILE)

points.columns   = [c.strip().lower() for c in points.columns]
clusters.columns = [c.strip().lower() for c in clusters.columns]

req_points   = {'cluster','latitude','longitude'}
req_clusters = {'cluster','centroid_lat','centroid_lon','n_points'}
missing_p = req_points - set(points.columns)
missing_c = req_clusters - set(clusters.columns)
if missing_p:
    st.error(f"Faltam colunas em points: {missing_p}"); st.stop()
if missing_c:
    st.error(f"Faltam colunas em clusters: {missing_c}"); st.stop()

# radius_km (p90) se não existir
if 'radius_km' not in clusters.columns:
    tmp = points.merge(clusters[['cluster','centroid_lat','centroid_lon']],
                       on='cluster', how='left', validate='m:1')
    tmp['dist_km'] = haversine_km(tmp['latitude'], tmp['longitude'],
                                  tmp['centroid_lat'], tmp['centroid_lon'])
    clusters = clusters.merge(
        tmp.groupby('cluster')['dist_km'].quantile(0.9).rename('radius_km').reset_index(),
        on='cluster', how='left'
    )

if 'region_macro' not in clusters.columns:
    clusters['region_macro'] = 'Indefinido'
if 'cd_name' not in clusters.columns:
    clusters['cd_name'] = clusters['cluster'].apply(lambda c: f"CD – Cluster {int(c):02d}")
clusters['radius_m'] = (clusters['radius_km'].fillna(0)*1000).astype(float)

# metadados no points
points = points.merge(clusters[['cluster','cd_name','region_macro']],
                      on='cluster', how='left')

# ---------------- Sidebar / filtros ----------------
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

show_heatmap   = st.sidebar.checkbox("Mostrar heatmap", value=False)
auto_fetch_uf  = st.sidebar.checkbox("Baixar UF GeoJSON automaticamente (se faltar)", value=False)
show_uf_layer  = st.sidebar.checkbox("Mostrar limites de UF (GeoJSON)", value=True)
choropleth_uf  = st.sidebar.checkbox(
    "Colorir UFs por densidade (GeoPandas)", value=False
) if GEOPANDAS_OK else False

# baixa automaticamente se pedido
if auto_fetch_uf and not UF_GEOJSON.exists():
    ok, msg = ensure_uf_geojson()
    st.toast(f"GeoJSON de UF: {msg}" if ok else f"Falhou ao baixar: {msg}")

# ---------------- Filtragem + limpeza ----------------
points_f   = points[points['cluster'].astype(int).isin(sel_clusters)].copy()
clusters_f = clusters[clusters['cluster'].astype(int).isin(sel_clusters)].copy()
if sel_regions:
    clusters_f = clusters_f[clusters_f['region_macro'].isin(sel_regions)]
    points_f   = points_f[points_f['cluster'].isin(clusters_f['cluster'])]

points_f   = clean_geo(points_f,   "longitude",    "latitude")
clusters_f = clean_geo(clusters_f, "centroid_lon", "centroid_lat")

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clusters", int(clusters_f['cluster'].nunique()))
c2.metric("Pontos",   int(points_f.shape[0]))
c3.metric("Raio p90 médio (km)", f"{clusters_f['radius_km'].mean():.2f}")
c4.metric("Maior raio p90 (km)", f"{clusters_f['radius_km'].max():.2f}")

# ---------------- Mapa ----------------
st.subheader("Mapa de Clusters e Áreas de Cobertura (p90)")

layers = []
if not clusters_f.empty:
    # auto-zoom DEPOIS da limpeza
    lons = pd.concat([points_f['longitude'], clusters_f['centroid_lon']], ignore_index=True)
    lats = pd.concat([points_f['latitude'],  clusters_f['centroid_lat']],  ignore_index=True)
    view = compute_view_safe(lons, lats)

    # 1) UF boundaries / choropleth
    if UF_GEOJSON.exists():
        if choropleth_uf and GEOPANDAS_OK:
            try:
                # GDF de UF (sem fiona)
                gdf_uf  = load_uf_geojson_gdf(UF_GEOJSON)
                gdf_pts = gpd.GeoDataFrame(
                    points_f,
                    geometry=gpd.points_from_xy(points_f["longitude"], points_f["latitude"]),
                    crs="EPSG:4326",
                )
                joined = gpd.sjoin(gdf_pts, gdf_uf[["uf_col", "geometry"]], how="left", predicate="within")
                resumo = (joined.drop(columns="geometry")
                                .groupby("uf_col").size()
                                .reset_index(name="num_pontos"))
                gdf_uf = gdf_uf.merge(resumo, on="uf_col", how="left").fillna({"num_pontos": 0})
                vmin, vmax = gdf_uf["num_pontos"].min(), gdf_uf["num_pontos"].max()
                gdf_uf["rgb"] = gdf_uf["num_pontos"].apply(lambda v: choropleth_color(v, vmin, vmax))
                gj_colored = json.loads(gdf_uf.to_json())
                layers.append(pdk.Layer(
                    "GeoJsonLayer",
                    data=gj_colored,
                    stroked=True, filled=True,
                    get_fill_color="properties.rgb",
                    get_line_color=[50, 50, 50, 200],
                    line_width_min_pixels=1,
                    pickable=True,
                ))
                st.caption(f"Choropleth UF: {int(vmin)}–{int(vmax)} pontos")
            except Exception as e:
                st.warning(f"Choropleth por UF não pôde ser gerado: {e}")
                # fallback para só contornos
                show_uf_layer = True

        if show_uf_layer and not choropleth_uf:
            try:
                with open(UF_GEOJSON, "r", encoding="utf-8") as f:
                    gj_data = json.load(f)
                layers.append(pdk.Layer(
                    "GeoJsonLayer",
                    data=gj_data,
                    stroked=True, filled=False,
                    get_line_color=[80, 80, 80, 180],
                    line_width_min_pixels=1
                ))
            except Exception as e:
                st.warning(f"Falha ao ler GeoJSON de UF: {e}")

    # 2) Heatmap
    if show_heatmap and not points_f.empty:
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=points_f,
            get_position='[longitude, latitude]',
            aggregation='"SUM"',
            get_weight=1,
            radiusPixels=30
        ))

    # 3) pontos (em pixels — visíveis em qualquer zoom)
    if show_points and not points_f.empty:
        pts = points_f if len(points_f) <= max_points else points_f.sample(max_points, random_state=42)
        if color_points_by_region:
            pts = pts.copy()
            pts['rgb'] = pts['region_macro'].apply(color_by_region)
            point_color = 'rgb'
        else:
            point_color = [30, 144, 255, 200]  # DodgerBlue

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position='[longitude, latitude]',
            get_fill_color=point_color,
            get_radius=6,                  # PIXELS
            radius_units="pixels",
            pickable=True,
            stroked=True,
            get_line_color=[0, 0, 0, 120],
            line_width_min_pixels=1
        ))

    # 4) áreas p90 (em metros — grandes)
    if show_areas and not clusters_f.empty:
        areas = clusters_f.assign(radius_m=clusters_f['radius_m'].clip(lower=200)).copy()
        areas['rgb'] = areas['region_macro'].apply(color_by_region)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=areas,
            get_position='[centroid_lon, centroid_lat]',
            get_radius="radius_m",          # METROS
            radius_units="meters",
            radius_min_pixels=3,
            pickable=True,
            filled=True,
            get_fill_color='rgb',
            stroked=True,
            get_line_color=[50, 50, 50, 220],
            line_width_min_pixels=1
        ))

    # 5) centróides
    if show_centroids and not clusters_f.empty:
        centers = clusters_f.copy()
        centers['rgb'] = centers['region_macro'].apply(color_by_region)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=centers,
            get_position='[centroid_lon, centroid_lat]',
            get_radius=110,
            radius_units="meters",
            get_fill_color='rgb',
            pickable=True
        ))

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

# ---------------- Tabelas (UF) ----------------
if choropleth_uf and GEOPANDAS_OK and UF_GEOJSON.exists() and not points_f.empty:
    try:
        gdf_uf  = load_uf_geojson_gdf(UF_GEOJSON)
        gdf_pts = gpd.GeoDataFrame(
            points_f,
            geometry=gpd.points_from_xy(points_f["longitude"], points_f["latitude"]),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(gdf_pts, gdf_uf[["uf_col", "geometry"]], how="left", predicate="within")
        resumo_uf = (joined.drop(columns="geometry")
                           .groupby("uf_col")
                           .size()
                           .reset_index(name="num_pontos")
                           .sort_values("num_pontos", ascending=False))
        st.markdown("### Pontos por UF (GeoPandas)")
        st.dataframe(resumo_uf, use_container_width=True)
    except Exception as e:
        st.warning(f"GeoPandas: não foi possível agregar por UF: {e}")

# ---------------- Tabela + download (clusters) ----------------
st.markdown("### Clusters selecionados")
st.dataframe(
    clusters_f[['cluster','cd_name','region_macro','radius_km','n_points']].sort_values('cluster'),
    use_container_width=True
)

csv = clusters_f.to_csv(index=False).encode("utf-8")
st.download_button("Baixar resumo de clusters (CSV)",
                   data=csv, file_name="clusters_summary_filtrado.csv", mime="text/csv")
