# Graph-Theory

Applied graph theory projects — currently one case study, with room to grow into other
graph algorithms (traversal, shortest paths, network flows) over time.

## Contents

### Electric Vehicle Charging Network

Graph model of Bogotá's EV charging network: builds a city graph and an aggregated
antenna graph, then compares routing algorithms (AOC vs. Dijkstra) across simulated
vehicle trips.

- `ev_charging_graph_model.ipynb` — main notebook (graph construction + routing comparison)
- `ev_model/` — supporting modules: entities, graph construction, routing, simulation, analysis, viz
- `Outputs/analysis/` — articulation points, bridges, centralities, service areas, and plots
  (betweenness vs. visits, route length distribution, runtimes, vehicles per antenna)
- Full write-up (Spanish): `Modelamiento de autos eléctricos, puntos de carga y conexión con la red.pdf`

## Technologies
- Python — NetworkX, Matplotlib, ipywidgets

## License
MIT — see [LICENSE](LICENSE).
