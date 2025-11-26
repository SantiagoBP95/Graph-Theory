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
import os
import ast
import pandas as pd
from collections import Counter


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
    else:
        # Ensure the provided `pos` mapping has positions for all nodes in G.
        # If some nodes are missing (e.g. due to a different graph instance or
        # label normalization mismatch), recompute a layout for the full graph
        # to avoid NetworkX KeyError when drawing.
        missing = [n for n in G.nodes() if n not in pos]
        if missing:
            print(f"Warning: missing positions for {len(missing)} nodes; recomputing layout.")
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

    plt.title(f"Vehicle {row.get('placa')} — AOC {row.get('aoc_time', 0.0):.4f}s — Dijkstra {row.get('dijkstra_time', 0.0):.4f}s")
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
    # Forward pos and layout_seed as positional args to avoid accidental
    # duplicate binding when callers pass `pos` both positionally and
    # via keywords (e.g., through widget wrappers). Using positional
    # arguments here prevents "multiple values for argument 'pos'" TypeErrors.
    return draw_vehicle(G, row, estaciones, rep_map, rep_to_ga, pos, layout_seed)

def _ensure_plots_dir(path: str = 'outputs/analysis/plots'):
    os.makedirs(path, exist_ok=True)
    return path


def _load_routes_df(df=None, csv_path: str = 'outputs/results_routes.csv'):
    if df is not None:
        return df
    if os.path.exists(csv_path):
        d = pd.read_csv(csv_path)
        # try to literal_eval route lists
        for col in ('aoc_route', 'dijkstra_route'):
            if col in d.columns:
                def _safe(x):
                    try:
                        return ast.literal_eval(x) if isinstance(x, str) else x
                    except Exception:
                        return x
                d[col] = d[col].apply(_safe)
        return d
    raise FileNotFoundError(f'Could not find routes CSV at: {csv_path}')


def plot_runtimes_boxplot(df=None, csv_path: str = 'outputs/results_routes.csv', save_path: str = None, log_scale: bool = True):
    d = _load_routes_df(df, csv_path)
    _ensure_plots_dir(os.path.dirname(save_path) if save_path else None or 'outputs/analysis/plots')
    times = d[[c for c in ('aoc_time', 'dijkstra_time') if c in d.columns]]
    if times.empty:
        raise ValueError('No runtime columns found in dataframe')
    plt.figure(figsize=(6, 4))
    data = times.melt(var_name='method', value_name='time')
    plt.boxplot([data.loc[data['method'] == m, 'time'].dropna() for m in data['method'].unique()], labels=data['method'].unique())
    if log_scale:
        plt.yscale('log')
    plt.title('Per-route runtime: AOC vs Dijkstra')
    plt.tight_layout()
    if not save_path:
        save_path = 'outputs/analysis/plots/runtimes_box.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_route_length_kde(df=None, csv_path: str = 'outputs/results_routes.csv', save_path: str = None):
    d = _load_routes_df(df, csv_path)
    _ensure_plots_dir(os.path.dirname(save_path) if save_path else None or 'outputs/analysis/plots')
    a_col = 'aoc_route_len' if 'aoc_route_len' in d.columns else None
    d_col = 'dijkstra_route_len' if 'dijkstra_route_len' in d.columns else None
    plt.figure(figsize=(6, 4))
    if a_col:
        vals = d[a_col].dropna()
        if len(vals) > 0:
            plt.hist(vals, bins=30, alpha=0.5, density=True, label='AOC')
    if d_col:
        vals = d[d_col].dropna()
        if len(vals) > 0:
            plt.hist(vals, bins=30, alpha=0.5, density=True, label='Dijkstra')
    plt.legend()
    plt.xlabel('route length (hops)')
    plt.title('Route length distribution')
    plt.tight_layout()
    if not save_path:
        save_path = 'outputs/analysis/plots/route_length_hist.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def _compute_visit_counts_from_df(d):
    visits = Counter()
    for col in ('aoc_route', 'dijkstra_route'):
        if col in d.columns:
            for r in d[col].dropna():
                if isinstance(r, (list, tuple)):
                    visits.update(r)
    return pd.DataFrame.from_records(list(visits.items()), columns=['node', 'visits']).sort_values('visits', ascending=False)


def plot_betweenness_vs_visits(centralities_csv: str = 'outputs/analysis/centralities.csv', routes_df=None, routes_csv: str = 'outputs/results_routes.csv', save_path: str = None):
    if not os.path.exists(centralities_csv):
        raise FileNotFoundError(f'Centralities CSV not found: {centralities_csv}')
    cent = pd.read_csv(centralities_csv)
    if routes_df is None:
        routes_df = _load_routes_df(None, routes_csv)
    visits_df = _compute_visit_counts_from_df(routes_df)
    merged = cent.merge(visits_df, left_on='node', right_on='node', how='left').fillna(0)
    _ensure_plots_dir(os.path.dirname(save_path) if save_path else None or 'outputs/analysis/plots')
    plt.figure(figsize=(6, 4))
    plt.scatter(merged['betweenness'], merged['visits'], alpha=0.7)
    plt.xlabel('betweenness')
    plt.ylabel('route visit count')
    plt.title('Betweenness vs route visit frequency')
    plt.tight_layout()
    if not save_path:
        save_path = 'outputs/analysis/plots/betweenness_vs_visits.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_vehicles_per_antenna(df=None, csv_path: str = 'outputs/results_routes.csv', save_path: str = None, top_n: int = 20):
    d = _load_routes_df(df, csv_path)
    if 'nearest_antena' not in d.columns:
        raise ValueError('nearest_antena column not found in routes dataframe')
    counts = d['nearest_antena'].value_counts().nlargest(top_n)
    _ensure_plots_dir(os.path.dirname(save_path) if save_path else None or 'outputs/analysis/plots')
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar')
    plt.ylabel('vehicles')
    plt.title('Vehicles per nearest antenna (top {})'.format(top_n))
    plt.tight_layout()
    if not save_path:
        save_path = 'outputs/analysis/plots/vehicles_per_antenna.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_dist_to_antenna_vs_length(df=None, routes_csv: str = 'outputs/results_routes.csv', save_path: str = None):
    d = _load_routes_df(df, routes_csv)
    if 'dist_to_antena' not in d.columns:
        raise ValueError('dist_to_antena column not found in routes dataframe')
    _ensure_plots_dir(os.path.dirname(save_path) if save_path else None or 'outputs/analysis/plots')
    plt.figure(figsize=(6, 4))
    if 'aoc_route_len' in d.columns:
        plt.scatter(d['dist_to_antena'], d['aoc_route_len'], alpha=0.5, label='AOC length')
    if 'dijkstra_route_len' in d.columns:
        plt.scatter(d['dist_to_antena'], d['dijkstra_route_len'], alpha=0.5, label='Dijkstra length')
    plt.xlabel('distance to antenna')
    plt.ylabel('route length (hops)')
    plt.legend()
    plt.title('Distance to antenna vs route length')
    plt.tight_layout()
    if not save_path:
        save_path = 'outputs/analysis/plots/dist_to_antena_vs_length.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_network_overlay(G: nx.Graph = None, cent_df: pd.DataFrame = None, routes_df=None, routes_csv: str = 'outputs/results_routes.csv', pos=None, top_n: int = 5, save_path: str = None):
    if G is None:
        raise ValueError('G (graph) is required for network overlay')
    if routes_df is None:
        try:
            routes_df = _load_routes_df(None, routes_csv)
        except Exception:
            routes_df = None
    visits_df = _compute_visit_counts_from_df(routes_df) if routes_df is not None else pd.DataFrame(columns=['node','visits'])
    top_nodes = visits_df.head(top_n)['node'].tolist() if not visits_df.empty else []
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    _ensure_plots_dir(os.path.dirname(save_path) if save_path else None or 'outputs/analysis/plots')
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color='lightgray', node_size=100)
    if top_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_color='red', node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title('Network: top visited nodes highlighted')
    plt.axis('off')
    plt.tight_layout()
    if not save_path:
        save_path = 'outputs/analysis/plots/network_top_visits.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


__all__.extend(['get_positions', 'draw_vehicle', 'draw_vehicle_by_index', 'plot_runtimes_boxplot', 'plot_route_length_kde', 'plot_betweenness_vs_visits', 'plot_vehicles_per_antenna', 'plot_dist_to_antenna_vs_length', 'plot_network_overlay'])
