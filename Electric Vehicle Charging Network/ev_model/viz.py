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


def get_positions(G: nx.Graph, seed: Optional[int] = 42):
    """Return a spring layout position mapping for `G`.

    This helper centralizes layout computation so notebooks and other callers
    can reuse a single layout for multiple figures.
    """
    return nx.spring_layout(G, seed=seed)


def draw_vehicle(
    G: nx.Graph,
    row: dict,
    estaciones,
    rep_map: dict,
    rep_to_ga: dict,
    pos=None,
    layout_seed: Optional[int] = 42,
    aoc_color: str = 'orange',
    dij_color: str = 'blue',
    vehicle_color: str = 'red',
    station_color: str = 'green',
    antenna_color: str = 'purple',
    nearest_color: str = 'magenta',
    figsize=(9, 6),
    node_size: int = 300,
):
    """Draw a vehicle's AOC vs Dijkstra routes and highlight stations/antennas.

    Parameters:
    - G: graph
    - row: dictionary with keys 'ubicacion','aoc_route','dijkstra_route','aoc_time','dijkstra_time'
    - estaciones: object with .total iterable of station entities (each with .ubicacion)
    - rep_map: mapping ga_label -> representative node (or None)
    - rep_to_ga: inverse mapping representative node -> ga_label
    - pos: optional positions mapping. If None, computed with layout_seed.
    """

    if pos is None:
        pos = nx.spring_layout(G, seed=layout_seed)

    aoc_r = row.get('aoc_route') or [row.get('ubicacion')]
    dij_r = row.get('dijkstra_route') or [row.get('ubicacion')]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)

    def _route_edges(route):
        if not isinstance(route, (list, tuple)) or len(route) < 2:
            return []
        return list(zip(route[:-1], route[1:]))

    aoc_edges = _route_edges(aoc_r)
    dij_edges = _route_edges(dij_r)

    if aoc_edges:
        nx.draw_networkx_edges(G, pos, edgelist=aoc_edges, edge_color=aoc_color, width=3.0, alpha=0.9)
    if dij_edges:
        nx.draw_networkx_edges(G, pos, edgelist=dij_edges, edge_color=dij_color, width=3.0, alpha=0.9)

    # vehicle node
    nx.draw_networkx_nodes(G, pos, nodelist=[row.get('ubicacion')], node_color=vehicle_color, node_size=160)

    # stations
    station_nodes = [e.ubicacion for e in getattr(estaciones, 'total', [])]
    if station_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=station_nodes, node_color=station_color, node_size=200, alpha=0.7)

    # antenna representative nodes (squares, purple)
    ant_nodes = [v for v in rep_map.values() if v is not None]
    if ant_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=ant_nodes, node_color=antenna_color, node_shape='s', node_size=260, alpha=0.5)

    # nearest antenna rep (magenta square)
    nearest_label = row.get('nearest_antena')
    nearest_rep = None
    if nearest_label:
        nearest_rep = rep_map.get(nearest_label)
    if nearest_rep:
        nx.draw_networkx_nodes(G, pos, nodelist=[nearest_rep], node_color=nearest_color, node_shape='s', node_size=320, alpha=0.9)

    plt.title(f"Vehículo {row.get('placa')} — AOC {row.get('aoc_time', 0.0):.4f}s — Dijkstra {row.get('dijkstra_time', 0.0):.4f}s")
    handles = [plt.Line2D([0], [0], color=aoc_color, lw=3), plt.Line2D([0], [0], color=dij_color, lw=3), plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=vehicle_color, markersize=8), plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=station_color, markersize=8), plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=antenna_color, markersize=10), plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=nearest_color, markersize=10)]
    labels = ['AOC route', 'Dijkstra route', 'Vehicle', 'Stations', 'Antenna (rep)', 'Nearest antenna']
    plt.legend(handles, labels)
    plt.axis('off')
    plt.show()


def draw_vehicle_by_index(G, rows, estaciones, rep_map, rep_to_ga, idx: int = 0, pos=None, layout_seed: Optional[int] = 42):
    """Convenience wrapper: draw vehicle at `rows[idx]` using `draw_vehicle`.
    Keeps notebook code succinct by delegating drawing logic here.
    """
    if not rows:
        raise ValueError('rows is empty')
    if idx < 0 or idx >= len(rows):
        raise IndexError('idx out of range')
    row = rows[idx]
    return draw_vehicle(G, row, estaciones, rep_map, rep_to_ga, pos=pos, layout_seed=layout_seed)

__all__.extend(['get_positions', 'draw_vehicle', 'draw_vehicle_by_index'])
