"""Routing utilities.

This module provides a small Dijkstra implementation for plain adjacency
dict graphs and a lightweight dispatcher that uses NetworkX for NetworkX
graphs. The dispatcher returns a tuple (order, distances, predecessors).
"""
import networkx as nx
import heapq


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
