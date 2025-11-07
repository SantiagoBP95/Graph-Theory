"""Graph construction helpers (city graph, aggregated graph, mapa (line-graph)).

These functions provide small, reproducible graphs used by the notebook and
by tests. They intentionally keep dependencies minimal and use networkx.
"""
import networkx as nx
from .config import DEFAULT_EDGE_WEIGHT


def build_G(nodes=None, edges=None):
    """Build and return the city graph from provided node and edge lists.

    Parameters
    - nodes: iterable of node labels (required)
    - edges: iterable of (u, v, weight) tuples (required)
    """
    if nodes is None or edges is None:
        raise ValueError(
            "build_G requires 'nodes' and 'edges' to be provided. "
            "Define the node/edge lists in the notebook and pass them to "
            "build_G(nodes, edges)."
        )

    G = nx.Graph()
    # add nodes and weighted edges; callers should pass edges as (u, v, weight)
    G.add_nodes_from(list(nodes))
    G.add_weighted_edges_from(list(edges))
    return G


def build_Ga(nodes=None, edges=None, default_weight=None):
    """Build and return the hierarchical/aggregated graph Ga from provided lists.

    Parameters
    - nodes: iterable of node labels for Ga (required)
    - edges: iterable of (u, v) tuples (required)
    - default_weight: numeric weight to assign to every Ga edge. If None,
      uses DEFAULT_EDGE_WEIGHT from the module config.
    """
    if nodes is None or edges is None:
        raise ValueError(
            "build_Ga requires 'nodes' and 'edges' to be provided. "
            "Define them in the notebook and call build_Ga(nodes, edges, default_weight=...)."
        )

    Ga = nx.Graph()
    Ga.add_nodes_from(list(nodes))
    Ga.add_edges_from(list(edges))
    weight = DEFAULT_EDGE_WEIGHT if default_weight is None else default_weight
    # assign a weight attribute to every Ga edge for downstream algorithms
    for u, v in Ga.edges():
        Ga[u][v]['weight'] = weight
    return Ga


def build_mapa(Ga):
    """Build a 'mapa' graph derived from the aggregated graph Ga.
    The 'mapa' is defined as the line graph of Ga, where nodes represent edges
    of Ga and edges exist when two Ga edges share a common endpoint.
    
    - If ``Ga`` has edges, we return its line graph (nodes represent Ga edges)
      which is a compact way to visualise adjacency between Ga links.
    - If the line graph nodes are tuples (edge endpoints), we relabel them to
      short string labels for nicer plotting (e.g. "A-B").
    - If `Ga` has no edges, return an empty Graph.
    """
    if not isinstance(Ga, nx.Graph):
        raise TypeError("build_mapa expects a networkx.Graph as input")

    if Ga.number_of_edges() == 0:
        return nx.Graph()

    try:
        mapa = nx.line_graph(Ga)
    except Exception:
        # fallback: return a shallow copy of Ga if line_graph fails
        return Ga.copy()

    # Relabel tuple nodes (edges of Ga) to readable strings like 'u-v'
    mapping = {}
    for n in list(mapa.nodes()):
        if isinstance(n, tuple) and len(n) >= 2:
            mapping[n] = f"{n[0]}-{n[1]}"
        else:
            mapping[n] = str(n)

    if mapping:
        mapa = nx.relabel_nodes(mapa, mapping)

    return mapa

__all__ = ['build_G', 'build_Ga', 'build_mapa']
