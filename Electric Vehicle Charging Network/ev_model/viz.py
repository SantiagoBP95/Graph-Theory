"""Visualization helpers for graphs.

This module provides a single flexible helper `draw_graph` that supports:
- layout selection ('spring' or 'planar')
- automatic edge-width scaling from a numeric edge attribute (default 'weight')
- optional rendering of edge weight labels (formatted)
- common styling options (node_size, node_color, font_size, seed)

"""
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, Optional


def _compute_widths(weights, scale: Tuple[float, float] = (0.8, 4.0)):
    """Normalize numeric weights into a list of widths in the provided scale.

    If all weights are equal (or weights list is empty) returns a list of the
    midpoint width.
    """
    if not weights:
        return []
    # replace None with 0 for normalization
    nums = [float(w) if w is not None else 0.0 for w in weights]
    mn = min(nums)
    mx = max(nums)
    lo, hi = scale
    if mx == mn:
        mid = (lo + hi) / 2.0
        return [mid for _ in nums]
    denom = mx - mn
    return [lo + (hi - lo) * ((w - mn) / denom) for w in nums]


def draw_graph(
    G: nx.Graph,
    layout: str = 'spring',
    show_edge_weights: bool = False,
    weight_attr: str = 'weight',
    scale_width: Tuple[float, float] = (0.8, 4.0),
    node_size: int = 350,
    node_color: str = 'lightblue',
    font_size: int = 8,
    seed: Optional[int] = None,
    cmap=None,
    **kwargs,
):
    """Draw a NetworkX graph with a few handy defaults.

    Parameters
    - G: networkx graph
    - layout: 'spring' or 'planar'
    - show_edge_weights: if True, draw edge labels from `weight_attr`
    - weight_attr: edge attribute name containing numeric weight
    - scale_width: (min_width, max_width) for mapping numeric weights
    - node_size, node_color, font_size, seed: layout/styling parameters
    - cmap, **kwargs: forwarded to drawing functions where applicable
    """
    
    for _k in ('show_edge_weights', 'weight_attr', 'scale_width'):
        kwargs.pop(_k, None)

    if layout == 'spring':
        pos = nx.spring_layout(G, seed=seed)
    elif layout == 'planar':
        try:
            pos = nx.planar_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=seed)
    else:
        pos = nx.spring_layout(G, seed=seed)

    plt.figure(figsize=kwargs.pop('figsize', (8, 6)))

    # nodes
    # draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=font_size)

    # edges: build widths from weight_attr if requested
    edge_list = list(G.edges())
    weights = [G[u][v].get(weight_attr) for u, v in edge_list]
    widths = _compute_widths([w if isinstance(w, (int, float)) else None for w in weights], scale=scale_width)
    if not widths:
        # fallback single width
        widths = kwargs.pop('width', [1.5])[0] if isinstance(kwargs.get('width', None), list) else kwargs.pop('width', 1.5)
        # nx.draw_networkx_edges accepts a scalar width too; draw with fallback
        nx.draw_networkx_edges(G, pos, width=widths, edge_color=kwargs.pop('edge_color', 'gray'), alpha=kwargs.pop('alpha', 0.9))
    else:
        # When widths is a list, NetworkX maps each width to corresponding edge
        nx.draw_networkx_edges(G, pos, width=widths, edge_color=kwargs.pop('edge_color', 'gray'), alpha=kwargs.pop('alpha', 0.9))

    # optionally draw edge labels (weights)
    if show_edge_weights:
        edge_labels = { (u, v): G[u][v].get(weight_attr) for u, v in edge_list if G[u][v].get(weight_attr) is not None }
        if edge_labels:
            # format numbers if numeric
            edge_labels_fmt = {}
            for k, v in edge_labels.items():
                try:
                    edge_labels_fmt[k] = f"{float(v):.1f}"
                except Exception:
                    edge_labels_fmt[k] = str(v)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_fmt, font_size=max(6, font_size - 1))

    plt.title(kwargs.pop('title', 'Graph'))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


__all__ = ['draw_graph']


def draw_route(
    G: nx.Graph,
    route,
    pos=None,
    layout_seed: Optional[int] = 42,
    base_node_color: str = 'lightblue',
    base_edge_color: str = 'lightgray',
    route_node_color: str = 'red',
    route_edge_color: str = 'red',
    route_width: float = 3.5,
    route_node_size: int = 180,
    node_size: int = 300,
    font_size: int = 8,
    title: Optional[str] = None,
    weight_attr: str = 'weight',
    show_edge_weights: bool = False,
    **kwargs,
):
    """Draw a graph and highlight a single route (list of nodes).

    Parameters
    - G: networkx Graph
    - route: iterable/list of node labels representing the path (ordered)
    - pos: optional precomputed positions mapping {node: (x,y)}; if None a spring
      layout will be computed using layout_seed.
    - base_node_color/base_edge_color: colors for the background graph
    - route_node_color/route_edge_color: colors used to highlight the route
    - route_width: width for the highlighted route edges
    - route_node_size: size for highlighted route nodes
    - node_size: size for background nodes
    - font_size: label font size
    - title: optional plot title
    - weight_attr: if show_edge_weights True, uses this attr for labels
    - show_edge_weights: if True draw weight labels for all edges
    - **kwargs: forwarded to matplotlib/networkx drawing functions
    """

    # sanitize route
    if not route or len(route) < 2:
        raise ValueError('Route must contain at least two nodes')

    if pos is None:
        pos = nx.spring_layout(G, seed=layout_seed)

    plt.figure(figsize=kwargs.pop('figsize', (8, 6)))

    # Draw base graph lightly
    # draw background nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color=base_node_color, node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=font_size)
    # draw all edges in a subdued color to provide context
    all_edge_list = list(G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=all_edge_list, edge_color=base_edge_color, alpha=0.6)

    # Highlight route edges and nodes
    route_edges = list(zip(route[:-1], route[1:]))
    # Only keep edges that exist in G (safe)
    route_edges_existing = [e for e in route_edges if G.has_edge(e[0], e[1]) or G.has_edge(e[1], e[0])]
    if route_edges_existing:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=route_edges_existing,
            width=route_width,
            edge_color=route_edge_color,
            alpha=0.95,
        )

    # draw route nodes on top
    nx.draw_networkx_nodes(G, pos, nodelist=route, node_color=route_node_color, node_size=route_node_size)

    # optionally draw edge labels
    if show_edge_weights:
        edge_labels = { (u, v): G[u][v].get(weight_attr) for u, v in all_edge_list if G[u][v].get(weight_attr) is not None }
        if edge_labels:
            # format
            edge_labels_fmt = {}
            for k, v in edge_labels.items():
                try:
                    edge_labels_fmt[k] = f"{float(v):.1f}"
                except Exception:
                    edge_labels_fmt[k] = str(v)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_fmt, font_size=max(6, font_size - 1))

    plt.title(title or kwargs.pop('title', 'Route'))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


__all__.append('draw_route')


def compare_routes(
    G: nx.Graph,
    aoc_route,
    dij_route,
    pos=None,
    layout_seed: Optional[int] = 42,
    aoc_color: str = 'orange',
    dij_color: str = 'dodgerblue',
    title: Optional[str] = None,
    node_size: int = 220,
    route_node_size: int = 120,
    route_width: float = 3.5,
    figsize=(9, 6),
):
    """Draw the graph and overlay two routes for comparison.

    Parameters
    - G: networkx Graph
    - aoc_route: ordered list of nodes for the AOC route
    - dij_route: ordered list of nodes for the Dijkstra route
    - pos: optional layout positions mapping; if None, computed with spring_layout(layout_seed)
    - aoc_color/dij_color: colors for the two routes
    - title: optional plot title
    - node_size/route_node_size/route_width: styling parameters
    - figsize: figure size
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=layout_seed)

    plt.figure(figsize=figsize)
    # base graph
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.6)

    def _draw_path(path, color):
        if not isinstance(path, (list, tuple)) or len(path) < 2:
            return
        eds = list(zip(path[:-1], path[1:]))
        eds_existing = [e for e in eds if G.has_edge(e[0], e[1]) or G.has_edge(e[1], e[0])]
        if eds_existing:
            nx.draw_networkx_edges(G, pos, edgelist=eds_existing, edge_color=color, width=route_width, alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=color, node_size=route_node_size)

    _draw_path(aoc_route, aoc_color)
    _draw_path(dij_route, dij_color)

    plt.title(title or 'AOC vs Dijkstra')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


__all__.append('compare_routes')
