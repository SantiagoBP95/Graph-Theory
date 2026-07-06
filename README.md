# Graph-Theory

Applied graph theory projects — currently one case study, with room to grow into other
graph algorithms (traversal, shortest paths, network flows) over time.

## Contents

### Electric Vehicle Charging Network

Graph model of Bogotá's EV charging network: builds a weighted city graph (neighborhoods
as nodes, roads as edges) and an aggregated antenna graph (charging-station topology),
then compares routing algorithms (ACO vs. Dijkstra) across simulated vehicle trips,
with centrality and critical-structure analysis (articulation points, bridges).

- `ev_charging_graph_model.ipynb` — main notebook (graph construction + routing comparison)
- `ev_model/` — supporting modules: entities, graph construction, routing, simulation, analysis, viz
- `Outputs/analysis/` — articulation points, bridges, centralities, service areas, and plots
  (betweenness vs. visits, route length distribution, runtimes, vehicles per antenna)
- Full write-up (Spanish): `Modelamiento de autos eléctricos, puntos de carga y conexión con la red.pdf`

## Setup

```bash
cd "Electric Vehicle Charging Network"
pip install -r requirements.txt
jupyter notebook ev_charging_graph_model.ipynb
```

## Technologies
- Python — NetworkX, Matplotlib, ipywidgets

## License
MIT — see [LICENSE](LICENSE).
