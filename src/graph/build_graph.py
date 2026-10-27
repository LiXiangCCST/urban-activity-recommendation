"""Build a heterogeneous urban graph from areas and resources."""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd


def _coord(row: pd.Series) -> tuple[float, float]:
    x = row["x"] if "x" in row and pd.notna(row["x"]) else row.get("centroid_x", 0.0)
    y = row["y"] if "y" in row and pd.notna(row["y"]) else row.get("centroid_y", 0.0)
    return float(x), float(y)


def _looks_like_integer_grid(areas: pd.DataFrame) -> bool:
    if not {"x", "y"}.issubset(areas.columns):
        return False
    return bool(((areas["x"] % 1 == 0) & (areas["y"] % 1 == 0)).all())


def _add_area_adjacency(graph: nx.Graph, areas: pd.DataFrame, k_neighbors: int = 4) -> None:
    if _looks_like_integer_grid(areas):
        area_by_coord = {(int(row["x"]), int(row["y"])): row["area_id"] for _, row in areas.iterrows()}
        for _, area in areas.iterrows():
            x_coord = int(area["x"])
            y_coord = int(area["y"])
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = area_by_coord.get((x_coord + dx, y_coord + dy))
                if neighbor:
                    graph.add_edge(area["area_id"], neighbor, edge_type="area_adjacency", weight=1.0)
        return

    area_rows = list(areas.iterrows())
    for _, area in area_rows:
        ax, ay = _coord(area)
        distances = []
        for _, other in area_rows:
            if area["area_id"] == other["area_id"]:
                continue
            ox, oy = _coord(other)
            distances.append((math.hypot(ax - ox, ay - oy), other["area_id"]))
        for distance, neighbor_id in sorted(distances)[:k_neighbors]:
            graph.add_edge(area["area_id"], neighbor_id, edge_type="area_adjacency", weight=float(distance))


def build_urban_graph(areas: pd.DataFrame, resources: pd.DataFrame, travel_times: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()

    for _, area in areas.iterrows():
        graph.add_node(area["area_id"], node_type="area", **area.to_dict())

    for _, resource in resources.iterrows():
        graph.add_node(resource["resource_id"], node_type="resource", **resource.to_dict())

    _add_area_adjacency(graph, areas)

    for _, row in travel_times.iterrows():
        edge_attrs = {
            "edge_type": "area_resource",
            "weight": float(row.get("walk_time", row.get("travel_time_minutes", 0.0))),
            "walk_time": float(row.get("walk_time", row.get("travel_time_minutes", 0.0))),
            "transit_time": float(row.get("transit_time", row.get("walk_time", row.get("travel_time_minutes", 0.0)))),
            "car_time": float(row.get("car_time", row.get("walk_time", row.get("travel_time_minutes", 0.0)))),
        }
        if row.get("category") == "transit":
            edge_attrs["edge_type"] = "area_transit"
        graph.add_edge(row["area_id"], row["resource_id"], **edge_attrs)

    return graph


def area_neighbors(graph: nx.Graph, area_id: str) -> list[str]:
    return sorted(
        neighbor
        for neighbor in graph.neighbors(area_id)
        if graph.nodes[neighbor].get("node_type") == "area"
    )
