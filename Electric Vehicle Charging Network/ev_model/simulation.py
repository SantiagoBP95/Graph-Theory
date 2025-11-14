"""Simple experiment runner for the EV charging model.

This module provides a small, parameterizable runner that uses other modules.
The runner can accept pre-built graphs; it returns a
summary including per-vehicle chosen routes to charging stations.
"""
from .graphs import build_G, build_Ga
from .entities import Flota, Estaciones
import random


def run_example(n=10, G=None, Ga=None, ubicaciones=None, random_seed=None, compute_routes=True):
    """Run a small example simulation.

    Parameters
    - n: number of vehicles in the fleet
    - G: optional networkx Graph for the detailed city graph (if None defaults used)
    - Ga: optional hierarchical graph (if None defaults used)
    - ubicaciones: optional mapping {Ga_node: tuple(of G node locations)} used by Estaciones
    - random_seed: optional int to seed randomness for reproducibility
    - compute_routes: if True, compute a candidate route to a charging station for each vehicle

    Returns a dict with keys: 'G','Ga','flota','estaciones','routes' (routes is a list of per-vehicle summaries)
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Require the caller (notebook) to build and provide G and Ga.
    if G is None:
        raise ValueError("simulation.run_example requires a city graph 'G'. Build G in the notebook and pass it via run_example(G=G, ...).")
    if Ga is None:
        raise ValueError("simulation.run_example requires an aggregated graph 'Ga'. Build Ga in the notebook and pass it via run_example(Ga=Ga, ...).")

    # Build a minimal 'ubicaciones' mapping for Ga if not provided. Map every Ga node
    # to a sample node from G (tuple) so Estaciones can pick one location.
    if ubicaciones is None:
        nodes = list(G.nodes())
        ga_nodes = list(Ga.nodes())
        # reproducible por seed si fue pasado
        if random_seed is not None:
            random.seed(random_seed)
        # si hay suficientes nodos en G, sampleamos sin reemplazo
        if len(nodes) >= len(ga_nodes):
            sampled = random.sample(nodes, k=len(ga_nodes))
        else:
            # si no, asignamos aleatoriamente con reemplazo
            sampled = [random.choice(nodes) for _ in ga_nodes]
        ubicaciones = {gnode: (sampled[i],) for i, gnode in enumerate(ga_nodes)}

    # Create Estaciones first so we can avoid placing vehicles at station locations
    estaciones = Estaciones(Ga, ubicaciones)
    # collect station node locations and ask Flota to avoid them when placing vehicles
    station_nodes = [e.ubicacion for e in estaciones.total]
    flota = Flota(n, G=G, exclude_nodes=station_nodes)

    routes = []
    if compute_routes:
        for auto in flota.total:
            try:
                ruta = auto.AOC(10, estaciones, G=G)
            except Exception as e:
                ruta = [auto.ubicacion]
            routes.append({
                'placa': auto.placa,
                'ubicacion': auto.ubicacion,
                'bateria': auto.bateria,
                'route': ruta,
                'route_len': max(0, len(ruta) - 1)
            })

    return {'G': G, 'Ga': Ga, 'flota': flota, 'estaciones': estaciones, 'routes': routes}
