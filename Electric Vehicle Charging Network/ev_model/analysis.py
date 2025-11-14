"""Graph analysis helpers for EV charging experiments.

Provides:
- compute_centralities(G): compute common centrality measures
- service_areas(G, rep_map): multi-source Dijkstra -> distances, paths, nearest mapping
- critical_structures(G): articulation points and bridges
- resilience_curve(G, strategy='degree', steps=20): simple attack simulation

All functions are lightweight wrappers around NetworkX utilities and return
Python-native structures (dicts, lists) suitable for saving into CSVs or
DataFrames in notebooks.
"""
from typing import Dict, Any, Tuple, List, Optional
import networkx as nx
import pandas as pd


def compute_centralities(G: nx.Graph, weight: str = 'weight') -> Dict[str, Dict[Any, float]]:
    """Compute a set of centrality measures for graph `G`.

    Returns a dict mapping metric_name -> {node: score}.
    Metrics: degree, betweenness, closeness, eigenvector, pagerank
    """
    res: Dict[str, Dict[Any, float]] = {}
    # degree centrality (normalized)
    res['degree'] = nx.degree_centrality(G)
    # betweenness centrality (uses weights if present)
    try:
        res['betweenness'] = nx.betweenness_centrality(G, weight=weight, normalized=True)
    except Exception:
        res['betweenness'] = nx.betweenness_centrality(G, normalized=True)
    # closeness centrality (distance is weight)
    try:
        res['closeness'] = nx.closeness_centrality(G, distance=weight)
    except Exception:
        res['closeness'] = nx.closeness_centrality(G)
    # eigenvector centrality (use numpy solver)
    try:
        res['eigenvector'] = nx.eigenvector_centrality_numpy(G, weight=weight)
    except Exception:
        try:
            res['eigenvector'] = nx.eigenvector_centrality(G, max_iter=200, weight=weight)
        except Exception:
            res['eigenvector'] = {n: 0.0 for n in G.nodes()}
    # pagerank
    try:
        res['pagerank'] = nx.pagerank(G, weight=weight)
    except Exception:
        res['pagerank'] = nx.pagerank(G)
    return res


def centralities_to_dataframe(centrals: Dict[str, Dict[Any, float]]) -> pd.DataFrame:
    """Convert centrality dict-of-dicts into a pandas DataFrame.

    Rows are nodes; columns are metric names.
    """
    df = pd.DataFrame(centrals)
    df.index.name = 'node'
    df = df.reset_index()
    return df


def service_areas(G: nx.Graph, rep_map: Dict[str, Optional[str]], weight: str = 'weight') -> Tuple[Dict[Any, float], Dict[Any, List[Any]], Dict[Any, Optional[str]]]:
    """Compute multi-source Dijkstra from representative nodes.

    Parameters
    - G: networkx graph
    - rep_map: mapping ga_label -> representative_node (or None)

    Returns (distances, paths, nearest_ga_for_node)
    - distances: node -> distance to nearest rep
    - paths: node -> path (list of nodes) from nearest rep to node
    - nearest_map: node -> ga_label (or None)
    """
    rep_nodes = [rep for rep in rep_map.values() if rep is not None]
    if not rep_nodes:
        return {}, {}, {}
    distances, paths = nx.multi_source_dijkstra(G, rep_nodes, weight=weight)
    # inverse map rep_node -> ga_label
    rep_to_ga = {rep: ga for ga, rep in rep_map.items() if rep is not None}
    nearest_map: Dict[Any, Optional[str]] = {}
    for node, path in paths.items():
        rep = path[0] if path else None
        nearest_map[node] = rep_to_ga.get(rep) if rep else None
    # for nodes not in paths, set nearest to None
    for node in G.nodes():
        if node not in nearest_map:
            nearest_map[node] = None
    return distances, paths, nearest_map


def critical_structures(G: nx.Graph) -> Dict[str, List[Any]]:
    """Return articulation points and bridges of the graph.

    Returns a dict with keys 'articulation_points' and 'bridges'.
    """
    art = list(nx.articulation_points(G))
    br = list(nx.bridges(G))
    return {'articulation_points': art, 'bridges': br}


def resilience_curve(G: nx.Graph, strategy: str = 'degree', steps: int = 20) -> pd.DataFrame:
    """Compute a simple resilience curve by removing nodes and measuring the
    size of the largest connected component.

    Parameters
    - strategy: 'degree' (remove highest degree first) or 'random'
    - steps: number of points in the curve

    Returns a DataFrame with columns: fraction_removed, giant_fraction
    """
    import random
    G0 = G.copy()
    n = G0.number_of_nodes()
    if n == 0:
        return pd.DataFrame(columns=['fraction_removed', 'giant_fraction'])
    if strategy == 'degree':
        order = sorted(G0.nodes(), key=lambda v: G0.degree(v), reverse=True)
    else:
        order = list(G0.nodes())
        random.shuffle(order)
    results = []
    # remove in batches
    batch = max(1, n // steps)
    removed = 0
    for i in range(0, n, batch):
        to_remove = order[i:i+batch]
        G0.remove_nodes_from(to_remove)
        removed += len(to_remove)
        comps = [len(c) for c in nx.connected_components(G0)] if G0.number_of_nodes() > 0 else [0]
        giant = max(comps) if comps else 0
        results.append({'fraction_removed': removed / n, 'giant_fraction': giant / n})
    df = pd.DataFrame(results)
    return df
