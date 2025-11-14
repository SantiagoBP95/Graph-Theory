"""Routing utilities.

This module provides a small Dijkstra implementation for plain adjacency
dict graphs and a lightweight dispatcher that uses NetworkX for NetworkX
graphs. The dispatcher returns a tuple (order, distances, predecessors).
"""
import networkx as nx
import heapq
from typing import Dict, List, Tuple, Any, Optional


def dijkstra_dict(gdict, source):
    """Dijkstra implementation for adjacency-dict graphs.

    Accepts a mapping of the form: {node: {neighbor: weight, ...}, ...} and
    computes shortest-path distances and simple predecessor pointers from
    `source`.

    Returns
    - order: list of nodes sorted by distance (closest first)
    - dist: mapping node -> shortest distance
    - prev: mapping node -> predecessor on the shortest path (or None)
    """
    dist = {v: float('inf') for v in gdict}
    prev = {v: None for v in gdict}
    dist[source] = 0
    Q = [(0, source)]
    while Q:
        d, u = heapq.heappop(Q)
        if d > dist[u]:
            continue
        for v, w in gdict[u].items():
            alt = dist[u] + float(w)
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(Q, (alt, v))
    order = sorted(dist, key=dist.get)
    return order, dist, prev


def dijkstra(g, source):
    """Dispatch to an implementation depending on graph type.

    If `g` has a `.nodes` attribute we assume it is a NetworkX graph and
    use NetworkX's single-source Dijkstra. Otherwise fall back to the
    adjacency-dict implementation above.
    """
    if hasattr(g, 'nodes'):
        # networkx graph: use the built-in implementation for speed and
        # correctness; we only return an order and distances here.
        length = nx.single_source_dijkstra_path_length(g, source, weight='weight')
        order = sorted(length, key=length.get)
        prev = {}  # predecessors not built in this wrapper
        return order, length, prev
    else:
        return dijkstra_dict(g, source)


def build_rep_map_from_antenas(antenas_obj, ubicaciones: Dict[str, Tuple[str, ...]]) -> Dict[str, Optional[str]]:
    """Build a mapping from GA antenna label -> representative node in G.

    Parameters
    - antenas_obj: instance providing `.total` iterable of antenna objects
      where each antenna has attribute `nodo` (the GA label).
    - ubicaciones: mapping ga_label -> tuple/list of representative node names

    Returns a dict mapping ga_label -> representative_node_or_None
    """
    rep_map: Dict[str, Optional[str]] = {}
    for a in getattr(antenas_obj, 'total', []):
        ga_label = getattr(a, 'nodo', None)
        if ga_label is None:
            continue
        reps = ubicaciones.get(ga_label)
        rep_map[ga_label] = reps[0] if reps and len(reps) > 0 else None
    return rep_map


def precompute_nearest_antennas(G: nx.Graph, rep_map: Dict[str, Optional[str]]) -> Tuple[Dict[Any, float], Dict[Any, List[Any]]]:
    """Run a multi-source Dijkstra from representative nodes and return
    distances and paths mapping node -> (distance, path).

    Returns (distances, paths). If there are no representative nodes,
    returns empty dicts.
    """
    rep_nodes = [rep for rep in rep_map.values() if rep is not None]
    if not rep_nodes:
        return {}, {}
    distances, paths = nx.multi_source_dijkstra(G, rep_nodes, weight='weight')
    return distances, paths


def annotate_rows_with_antenna(rows: List[Dict[str, Any]], distances: Dict[Any, float], paths: Dict[Any, List[Any]], rep_to_ga: Dict[Any, str]) -> None:
    """Annotate each row in-place with 'nearest_antena' and 'dist_to_antena'.

    Parameters
    - rows: list of dict-like rows with key 'ubicacion'
    - distances: mapping node -> distance (from precompute)
    - paths: mapping node -> path (from precompute)
    - rep_to_ga: mapping representative_node -> ga_label
    """
    for row in rows:
        node = row.get('ubicacion')
        dist = distances.get(node, float('inf'))
        path = paths.get(node)
        nearest_rep = path[0] if path else None
        nearest_ga = rep_to_ga.get(nearest_rep) if nearest_rep else None
        row['nearest_antena'] = nearest_ga
        row['dist_to_antena'] = dist


def integrate_antennas(rows: List[Dict[str, Any]], G: nx.Graph, antenas_obj, ubicaciones: Dict[str, Tuple[str, ...]]):
    """High-level helper that builds rep_map, precomputes distances and
    annotates `rows` with nearest antenna info.

    Returns a tuple: (rep_map, rep_to_ga, distances, paths)
    """
    rep_map = build_rep_map_from_antenas(antenas_obj, ubicaciones)
    rep_to_ga = {rep: ga for ga, rep in rep_map.items() if rep is not None}
    distances, paths = precompute_nearest_antennas(G, rep_map)
    annotate_rows_with_antenna(rows, distances, paths, rep_to_ga)
    return rep_map, rep_to_ga, distances, paths
