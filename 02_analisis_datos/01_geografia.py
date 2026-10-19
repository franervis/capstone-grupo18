#!/usr/bin/env python3
"""Análisis geográfico reproducible para la red pública de sangre de Santiago.

No modifica los datos originales. Construye una red vial no dirigida y ponderada
por longitud física, genera matrices de control y red, tablas de diagnóstico y
figuras académicas en outputs/.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path

# Fuerza un backend no interactivo y una caché escribible en ejecuciones aisladas.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
from dataset_config import ORIGINAL_DATA, data_file, output_dir

DATA = ORIGINAL_DATA
OUT = output_dir("geografia")
GPKG = ORIGINAL_DATA / "chile.gpkg"
ROADS_LAYER = "gis_osm_roads_free"
CRS_GEO = "EPSG:4326"
CRS_METRIC = "EPSG:32719"  # WGS 84 / UTM 19S
BANK_NAME = "Centro Metropolitano de Sangre"

# Clases OSM apropiadas para vehículos. Se excluyen footway, path, cycleway,
# pedestrian, steps y tracks, pues podrían crear atajos no utilizables por la flota.
DRIVABLE = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "residential", "living_street", "service",
    "unclassified", "road",
}


def inspect_inputs() -> pd.DataFrame:
    """Inventaría CSV y capas del GeoPackage sin cargar capas masivas completas."""
    rows = []
    for path in sorted(DATA.glob("*.csv")):
        frame = pd.read_csv(path)
        rows.append({
            "archivo": path.name,
            "filas": len(frame),
            "columnas": len(frame.columns),
            "tipos": json.dumps({c: str(t) for c, t in frame.dtypes.items()}, ensure_ascii=False),
            "primeras_filas": frame.head(3).to_json(orient="records", force_ascii=False),
        })
    pd.DataFrame(rows).to_csv(OUT / "inspeccion_csv.csv", index=False)

    connection = sqlite3.connect(GPKG)
    layer_rows = []
    try:
        query = """
        SELECT c.table_name, c.data_type, g.geometry_type_name, c.srs_id,
               c.min_x, c.min_y, c.max_x, c.max_y
        FROM gpkg_contents c
        LEFT JOIN gpkg_geometry_columns g ON c.table_name = g.table_name
        ORDER BY c.table_name
        """
        for name, dtype, geom, srs, minx, miny, maxx, maxy in connection.execute(query):
            n = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            columns = [r[1] for r in connection.execute(f'PRAGMA table_info("{name}")')]
            layer_rows.append({
                "capa": name, "cantidad_geometrias": n, "tipo_geometria": geom,
                "crs_srs_id": srs, "columnas": ", ".join(columns),
                "min_x": minx, "min_y": miny, "max_x": maxx, "max_y": maxy,
            })
    finally:
        connection.close()
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(OUT / "inspeccion_capas_gpkg.csv", index=False)
    return layers


def validate_hospitals(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if len(df) != 23:
        issues.append(f"Se esperaban 23 ubicaciones y hay {len(df)}.")
    if (df["tipo"].str.casefold() == "banco").sum() != 1:
        issues.append("Debe existir exactamente un registro de tipo Banco.")
    if (df["tipo"].str.casefold() == "hospital").sum() != 22:
        issues.append("Debe haber exactamente 22 hospitales.")
    if df.isna().any().any():
        issues.append("Existen valores faltantes.")
    if df["nombre"].duplicated().any():
        issues.append("Existen nombres duplicados.")
    if df.duplicated(["lat", "lon"]).any():
        issues.append("Existen coordenadas exactamente duplicadas.")
    outside = ~df["lat"].between(-34.0, -33.0) | ~df["lon"].between(-71.0, -70.0)
    if outside.any():
        issues.append("Coordenadas fuera del rango amplio de Santiago: " + ", ".join(df.loc[outside, "nombre"]))
    if BANK_NAME not in set(df["nombre"]):
        issues.append(f"No se encontró el banco esperado: {BANK_NAME}.")
    return issues


def haversine_matrix(df: pd.DataFrame) -> pd.DataFrame:
    lat = np.radians(df["lat"].to_numpy())
    lon = np.radians(df["lon"].to_numpy())
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(distances, 0.0)
    return pd.DataFrame(distances, index=df["nombre"], columns=df["nombre"])


def pair_table(matrix: pd.DataFrame) -> pd.DataFrame:
    names = matrix.index.to_list()
    return pd.DataFrame([
        {"ubicacion_1": names[i], "ubicacion_2": names[j], "distancia_km": matrix.iat[i, j]}
        for i in range(len(names)) for j in range(i + 1, len(names))
    ])


def load_roads(hospitals: pd.DataFrame) -> gpd.GeoDataFrame:
    # Margen de ~11 km. El rectángulo incluye corredores alternativos entre todos
    # los puntos y evita cargar la red de todo Chile (1+ millón de segmentos).
    bbox = (hospitals.lon.min() - 0.12, hospitals.lat.min() - 0.12,
            hospitals.lon.max() + 0.12, hospitals.lat.max() + 0.12)
    roads = gpd.read_file(GPKG, layer=ROADS_LAYER, bbox=bbox, engine="pyogrio")
    roads = roads[roads["fclass"].isin(DRIVABLE) & roads.geometry.notna()].copy()
    roads = roads.to_crs(CRS_METRIC)
    roads = roads[~roads.geometry.is_empty & (roads.geometry.length > 0)].copy()
    roads["longitud_m"] = roads.geometry.length
    return roads


def build_graph(roads: gpd.GeoDataFrame) -> nx.Graph:
    """Crea aristas entre cada par consecutivo de vértices de cada LineString."""
    graph = nx.Graph()
    for geom in roads.geometry:
        if geom.geom_type == "MultiLineString":
            parts = geom.geoms
        else:
            parts = [geom]
        for part in parts:
            coords = list(part.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                a = (round(a[0], 3), round(a[1], 3))
                b = (round(b[0], 3), round(b[1], 3))
                if a == b:
                    continue
                length = math.hypot(b[0] - a[0], b[1] - a[1])
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] = min(graph[a][b]["weight"], length)
                else:
                    graph.add_edge(a, b, weight=length)
    # Evita snaps a pequeños componentes aislados (estacionamientos, caminos
    # incompletos). Todos los destinos deben pertenecer a la red metropolitana.
    largest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def snap_locations(hospitals: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    points = gpd.GeoDataFrame(hospitals.copy(), geometry=gpd.points_from_xy(hospitals.lon, hospitals.lat), crs=CRS_GEO)
    points = points.to_crs(CRS_METRIC)
    nodes = np.asarray(list(graph.nodes), dtype=float)
    tree = cKDTree(nodes)
    distances, indices = tree.query(np.column_stack([points.geometry.x, points.geometry.y]))
    result = hospitals[["nombre", "lat", "lon"]].copy()
    result["nodo_red"] = [f"{nodes[i, 0]:.3f},{nodes[i, 1]:.3f}" for i in indices]
    result["nodo_x"] = nodes[indices, 0]
    result["nodo_y"] = nodes[indices, 1]
    result["distancia_snap_m"] = distances
    return result


def network_matrix(names: list[str], snaps: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    nodes = list(zip(snaps["nodo_x"], snaps["nodo_y"]))
    matrix = np.full((len(nodes), len(nodes)), np.inf)
    for i, source in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        for j, target in enumerate(nodes):
            # Distancia puerta-a-puerta aproximada: acceso recto de cada ubicación
            # a su nodo más el camino por red. Incluir los conectores evita que el
            # snap subestime artificialmente la distancia física observada.
            access = snaps.iloc[i]["distancia_snap_m"] + snaps.iloc[j]["distancia_snap_m"]
            matrix[i, j] = (lengths.get(target, np.inf) + access) / 1000
    np.fill_diagonal(matrix, 0.0)
    return pd.DataFrame(matrix, index=names, columns=names)


def plot_base(roads: gpd.GeoDataFrame, hospitals: pd.DataFrame, snaps: pd.DataFrame) -> None:
    points = gpd.GeoDataFrame(hospitals.copy(), geometry=gpd.points_from_xy(hospitals.lon, hospitals.lat), crs=CRS_GEO).to_crs(CRS_METRIC)
    fig, ax = plt.subplots(figsize=(11, 10))
    roads.plot(ax=ax, color="#b8b8b8", linewidth=0.22, alpha=0.6)
    hosp = points[points.tipo == "Hospital"]
    bank = points[points.tipo == "Banco"]
    hosp.plot(ax=ax, color="#1769aa", markersize=28, label="Hospital", zorder=3)
    bank.plot(ax=ax, color="#c62828", marker="*", markersize=170, label="Centro Metropolitano", zorder=4)
    for _, row in points.iterrows():
        ax.annotate(row["nombre"], (row.geometry.x, row.geometry.y), xytext=(3, 3),
                    textcoords="offset points", fontsize=6.5)
    ax.set(title="Centro Metropolitano y hospitales sobre la red vial utilizada",
           xlabel="Este UTM 19S (m)", ylabel="Norte UTM 19S (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mapa_hospitales_red.png", dpi=220)
    plt.close(fig)


def plot_nearest(roads: gpd.GeoDataFrame, hospitals: pd.DataFrame, snaps: pd.DataFrame,
                 matrix: pd.DataFrame, graph: nx.Graph) -> None:
    hosp_names = hospitals.loc[hospitals.tipo == "Hospital", "nombre"].to_list()
    snap_by_name = snaps.set_index("nombre")
    fig, ax = plt.subplots(figsize=(11, 10))
    roads.plot(ax=ax, color="#d0d0d0", linewidth=0.2, alpha=0.55)
    drawn = set()
    for name in hosp_names:
        candidates = matrix.loc[name, hosp_names].drop(name)
        nearest = candidates.idxmin()
        pair = tuple(sorted((name, nearest)))
        if pair in drawn:
            continue
        drawn.add(pair)
        a = (snap_by_name.loc[name, "nodo_x"], snap_by_name.loc[name, "nodo_y"])
        b = (snap_by_name.loc[nearest, "nodo_x"], snap_by_name.loc[nearest, "nodo_y"])
        path = nx.shortest_path(graph, a, b, weight="weight")
        ax.plot([p[0] for p in path], [p[1] for p in path], color="#ef6c00", linewidth=1.3, alpha=0.9)
    pts = gpd.GeoDataFrame(hospitals, geometry=gpd.points_from_xy(hospitals.lon, hospitals.lat), crs=CRS_GEO).to_crs(CRS_METRIC)
    pts[pts.tipo == "Hospital"].plot(ax=ax, color="#1769aa", markersize=30, label="Hospital", zorder=3)
    pts[pts.tipo == "Banco"].plot(ax=ax, color="#c62828", marker="*", markersize=170,
                                  label="Centro Metropolitano", zorder=4)
    ax.set(title="Relaciones de cercanía: vecino hospitalario más próximo",
           xlabel="Este UTM 19S (m)", ylabel="Norte UTM 19S (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mapa_vecinos_mas_cercanos.png", dpi=220)
    plt.close(fig)


def geographic_summary(hospitals: pd.DataFrame, network: pd.DataFrame) -> pd.DataFrame:
    hospital_rows = hospitals[hospitals.tipo == "Hospital"].copy()
    names = hospital_rows.nombre.to_list()
    records = []
    for _, row in hospital_rows.iterrows():
        name = row.nombre
        ordered = network.loc[name, names].drop(name).sort_values()
        records.append({
            "nombre": name, "tamano": row.tamano, "capacidad_maxima": row.capacidad_maxima,
            "hospital_mas_cercano": ordered.index[0],
            "distancia_hospital_mas_cercano_km": ordered.iloc[0],
            "distancia_centro_metropolitano_km": network.loc[name, BANK_NAME],
            "distancia_promedio_otros_hospitales_km": ordered.mean(),
            "hospital_cercano_1": ordered.index[0], "distancia_cercano_1_km": ordered.iloc[0],
            "hospital_cercano_2": ordered.index[1], "distancia_cercano_2_km": ordered.iloc[1],
            "hospital_cercano_3": ordered.index[2], "distancia_cercano_3_km": ordered.iloc[2],
        })
    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    layers = inspect_inputs()
    hospitals = pd.read_csv(data_file("hospitales.csv"))
    issues = validate_hospitals(hospitals)
    gdf = gpd.GeoDataFrame(hospitals.copy(), geometry=gpd.points_from_xy(hospitals.lon, hospitals.lat), crs=CRS_GEO)

    geodesic = haversine_matrix(hospitals)
    geodesic.to_csv(OUT / "matriz_distancias_geodesicas_km.csv", float_format="%.6f")
    geo_pairs = pair_table(geodesic).sort_values("distancia_km")
    pd.concat([geo_pairs.head(5).assign(grupo="5 más cercanas"),
               geo_pairs.tail(5).sort_values("distancia_km", ascending=False).assign(grupo="5 más lejanas")]) \
      .to_csv(OUT / "extremos_distancias_geodesicas.csv", index=False)

    roads = load_roads(hospitals)
    graph = build_graph(roads)
    snaps = snap_locations(hospitals, graph)
    snaps.to_csv(OUT / "snap_hospitales_red.csv", index=False, float_format="%.3f")
    network = network_matrix(hospitals.nombre.to_list(), snaps, graph)
    network.to_csv(OUT / "matriz_distancias_red_km.csv", float_format="%.6f")

    pairs_geo = pair_table(geodesic).rename(columns={"distancia_km": "distancia_geodesica_km"})
    pairs_net = pair_table(network).rename(columns={"distancia_km": "distancia_red_km"})
    comparison = pairs_geo.merge(pairs_net, on=["ubicacion_1", "ubicacion_2"])
    comparison["ratio_red_geodesica"] = comparison.distancia_red_km / comparison.distancia_geodesica_km
    comparison.to_csv(OUT / "comparacion_distancias.csv", index=False, float_format="%.6f")

    ratios = comparison.ratio_red_geodesica.replace([np.inf, -np.inf], np.nan).dropna()
    ratio_stats = {"promedio": ratios.mean(), "mediana": ratios.median(),
                   "p95": ratios.quantile(.95), "maximo": ratios.max()}
    comparison.nlargest(10, "ratio_red_geodesica").to_csv(OUT / "ratios_altos_distancias.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(comparison.distancia_geodesica_km, comparison.distancia_red_km,
               s=23, alpha=.65, color="#1769aa", edgecolor="white", linewidth=.25)
    upper = max(comparison.distancia_geodesica_km.max(), comparison.distancia_red_km.replace(np.inf, np.nan).max()) * 1.04
    ax.plot([0, upper], [0, upper], "--", color="#c62828", label="y = x")
    ax.set(xlabel="Distancia geodésica (km)", ylabel="Distancia mínima por red (km)",
           title="Validación de distancias: geodésica versus red vial", xlim=(0, upper), ylim=(0, upper))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "validacion_distancias.png", dpi=220)
    plt.close(fig)

    summary = geographic_summary(hospitals, network)
    summary.to_csv(OUT / "resumen_geografico_hospitales.csv", index=False, float_format="%.6f")
    plot_base(roads, hospitals, snaps)
    plot_nearest(roads, hospitals, snaps, network, graph)

    bank_distances = summary[["nombre", "distancia_centro_metropolitano_km"]].sort_values("distancia_centro_metropolitano_km")
    report = {
        "capa_vial": ROADS_LAYER,
        "crs_original": "OGC:CRS84 (WGS 84, longitud/latitud; srs_id interno 100000)",
        "crs_distancias": CRS_METRIC,
        "red_dirigida": False,
        "motivo_red_no_dirigida": "Benchmark de distancia física y orden de visita; se posterga el sentido vial para el modelo operativo.",
        "convencion_distancia": "camino entre nodos + conectores rectos de snap en ambos extremos",
        "clases_viales_incluidas": sorted(DRIVABLE),
        "segmentos_viales_recortados_y_filtrados": len(roads),
        "nodos_componente_principal": graph.number_of_nodes(),
        "aristas_componente_principal": graph.number_of_edges(),
        "problemas_validacion_hospitales": issues,
        "snap_promedio_m": snaps.distancia_snap_m.mean(),
        "snap_maximo_m": snaps.distancia_snap_m.max(),
        "ubicacion_snap_maximo": snaps.loc[snaps.distancia_snap_m.idxmax(), "nombre"],
        "pares_no_conectados": int(np.isinf(network.to_numpy()).sum() / 2),
        "matriz_red_simetrica": bool(np.allclose(network, network.T, equal_nan=True)),
        "ratio_estadisticas": ratio_stats,
        "violaciones_red_menor_geodesica": int((comparison.ratio_red_geodesica < .99).sum()),
        "distancia_banco_hospital_min_km": bank_distances.distancia_centro_metropolitano_km.min(),
        "distancia_banco_hospital_max_km": bank_distances.distancia_centro_metropolitano_km.max(),
    }
    with open(OUT / "reporte_ejecucion.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    bank_distances.head(10).to_csv(OUT / "10_hospitales_mas_cercanos_al_centro.csv", index=False)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n10 hospitales más cercanos al Centro Metropolitano:")
    print(bank_distances.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
