import random
import math
import networkx as nx
import types
import time
from .routing import dijkstra


class Auto:
	"""Simple vehicle entity.

	Attributes
	- placa: identifier string
	- bateria: numeric battery level
	- ubicacion: current node in the city graph G

	Methods
	- AOC(n, E, G=None): perform a set of random-walk candidate searches for a charging station
	  among the stations in `E.total` and return the selected route (list of nodes).
	"""

	def __init__(self, placa, bateria, ubicacion):
		self.placa = placa
		self.bateria = bateria
		self.ubicacion = ubicacion

	def __repr__(self):
		return "({0},{1},{2})".format(self.placa, self.bateria, self.ubicacion)

	def AOC(self, n, E, G=None, max_steps=200, epsilon=1e-6):
		"""
		Randomized-walk search for a route to a charging station.
		- epsilon: small value to avoid division-by-zero for zero-weight edges.
		"""
		if G is None:
			raise ValueError(
				"G (global city graph) is not defined and was not passed to Auto.AOC"
			)

		E_nodos = [x.ubicacion for x in E.total]

		if self.ubicacion in E_nodos:
			return [self.ubicacion]

		best_score = -math.inf
		best_path = [self.ubicacion]

		for _ in range(n):
			path = [self.ubicacion]
			score_terms = []
			steps = 0

			while steps < max_steps and path[-1] not in E_nodos:
				current = path[-1]
				vecinos = [v for v in G.neighbors(current) if v not in path]

				if not vecinos:
					# camino truncado → descartado
					score_terms = []
					break

				nxt = random.choice(vecinos)
				w = G[current][nxt].get("weight", 1.0)

				# usar epsilon para evitar dividir entre 0 y no inflar scores
				inv = 1.0 / (w + epsilon)
				score_terms.append(inv)

				path.append(nxt)
				steps += 1

			if path[-1] in E_nodos and len(score_terms) > 0:
				# usar promedio de inversos (estable)
				score = sum(score_terms) / len(score_terms)
				# opcional: penalizar rutas muy largas
				# score = score / (1 + 0.01 * (len(path)-1))

				if score > best_score:
					best_score = score
					best_path = path

		return best_path

	def route_dijkstra(self, E, G=None, measure_time: bool = False):
		"""
		Find a route to the closest charging station using Dijkstra distances.
		Robust: use the (possibly custom) dijkstra dispatcher only to obtain
		distances; always reconstruct the final path with NetworkX shortest_path
		(weight='weight') to avoid different 'prev' formats.
		Returns route or (route, elapsed_time) if measure_time=True.
		"""
		if G is None:
			raise ValueError(
				"G (global city graph) is not defined and was not passed to route_dijkstra"
			)

		E_nodos = [x.ubicacion for x in E.total]

		if self.ubicacion in E_nodos:
			result = [self.ubicacion]
			return (result, 0.0) if measure_time else result

		if measure_time:
			start_time = time.perf_counter()

		# obtener distancias (intentar dispatcher, si falla usar NetworkX)
		try:
			_, dist, _ = dijkstra(G, self.ubicacion)
		except Exception:
			dist = nx.single_source_dijkstra_path_length(G, self.ubicacion, weight='weight')

		# dist debe ser dict-like
		if not hasattr(dist, "get"):
			# intentar convertir a dict si es un mapping no típico
			try:
				dist = dict(dist)
			except Exception:
				raise TypeError("dijkstra() returned an invalid dist mapping")

		# filtrar estaciones alcanzables y con distancia finita
		reachable = [s for s in E_nodos if (s in dist and dist.get(s, float('inf')) < float('inf'))]

		if not reachable:
			result = [self.ubicacion]
			return (result, 0.0) if measure_time else result

		# elegir la estación más cercana según distancias
		nearest = min(reachable, key=lambda s: dist.get(s, float('inf')))

		# reconstruir la ruta con NetworkX (garantiza camino mínimo según 'weight')
		try:
			route = nx.shortest_path(G, source=self.ubicacion, target=nearest, weight='weight')
		except Exception:
			# fallback defensivo: si falla, devolvemos solo la ubicación actual
			route = [self.ubicacion]

		if measure_time:
			elapsed = time.perf_counter() - start_time
			return route, elapsed

		return route


class Flota:
	"""Fleet container that builds `total` Auto instances.

	Constructor will create `total` vehicles and place them at random nodes
	on the provided graph `G`. If `G` is not provided the global `G` is
	attempted via `_get_global` (historical notebook compatibility).
	"""

	def __init__(self, total, G=None, exclude_nodes=None):
		"""Create `total` vehicles and place them at random nodes in `G`.

		Parameters
		- total: number of vehicles
		- G: networkx Graph with nodes to place vehicles on
		- exclude_nodes: optional iterable of node labels to avoid when placing vehicles
		"""
		if G is None:
			raise ValueError("G (global city graph) is not defined and was not passed to Flota.__init__")
		# prepare candidate nodes excluding any provided station nodes
		all_nodes = list(G.nodes)
		exclude_set = set(exclude_nodes) if exclude_nodes is not None else set()
		candidates = [n for n in all_nodes if n not in exclude_set]
		# if excluding leaves no candidates, fall back to all nodes (avoid crash)
		if not candidates:
			candidates = all_nodes
		f = []
		for i in range(total):
			placa = "car" + str(i)
			bateria = random.randrange(50, 100)
			ubicacion = random.choice(candidates)
			autoi = Auto(placa, bateria, ubicacion)
			f.append(autoi)
		self.total = f

	def __repr__(self):
		pf = ""
		for a in self.total:
			pf += repr(a)
		return pf


class Enrutador:
	"""Routing helper base class.

	Provides a helper `nodos_adyacentes` to return neighbors of a node that
	are not in a provided route list. This is used by higher-level classes.
	"""

	def __init__(self, enrutamiento=None, nodo=None):
		self.enrutamiento = enrutamiento
		self.nodo = nodo

	def nodos_adyacentes(self, r, nodo, Gr=None):
		"""Return neighbors of `nodo` in graph `Gr` that are not in route `r`.

		Parameters
		- r: iterable of nodes representing a route (to exclude)
		- nodo: node whose neighbors are queried
		- Gr: optional NetworkX graph; falls back to global `Gr` if not provided
		"""

		if Gr is None:
			raise ValueError("Gr (hierarchical graph) is not defined and was not passed to nodos_adyacentes")
		l = []
		vecinos = list(Gr.neighbors(nodo))
		if len(vecinos) == 0:
			return l
		for i in vecinos:
			if i not in r:
				l.append(i)
		return l


class Servidor(Enrutador):
	"""Lightweight server/antenna descriptor used by Antenas."""

	def __init__(self, ip, dtb, nodo, enrutamiento=None):
		super().__init__(enrutamiento=enrutamiento, nodo=nodo)
		self.ip = ip
		self.dtb = dtb

	def __repr__(self):
		return "ip: {0}, dtb: {1}".format(self.ip, self.dtb)


class Antenas(Enrutador):
	"""Container of antenna/server nodes built from a hierarchical graph `Ga`.

	The constructor creates a `Servidor` instance per Ga node and stores them
	in `.total`.
	"""

	def __init__(self, Ga=None):
		if Ga is None:
			raise ValueError("Ga (aggregated graph) is not defined and was not passed to Antenas.__init__")
		t = []
		s = 0
		enr = {}
		for i in Ga:
			enr[i] = [i, float("inf")]
			antena = Servidor(s, [], i, enr)
			s += 1
			t.append(antena)
		self.total = t

	def __repr__(self):
		r = ""
		for i in self.total:
			r += repr(i) + "\n"
		return r


class EstacionCarga(Enrutador):
	"""Represents a single charging station with queue/slots.

	Attributes: capacidad (int), cola (list), slots (list), ubicacion (node),
	nodo (node id), ip (identifier).
	"""

	def __init__(self, capacidad, cola, slots, ubicacion, nodo, ip, enrutamiento=None):
		super().__init__(enrutamiento=enrutamiento, nodo=nodo)
		self.cola = cola
		self.slots = slots
		self.ubicacion = ubicacion
		self.nodo = nodo
		self.ip = ip
		self.capacidad = capacidad
		self.enrutamiento = enrutamiento

	def __repr__(self):
		return "ip: {0}, nodo: {2}, Ubicacion: {1},  Slots disponibles: {3}, Autos en cola: {4}".format(self.ip, self.ubicacion, self.nodo, self.capacidad - len(self.slots), len(self.cola))


class Estaciones(EstacionCarga):
	"""Collection of charging stations built from a hierarchical graph `Gr`.

	The constructor expects a graph `Gr` and a mapping `ubicaciones` that
	provides candidate G-node locations for each Gr node. It validates the
	mapping and creates an `EstacionCarga` per Gr node stored in `.total`.
	"""

	def __init__(self, Gr=None, ubicaciones=None):

		if Gr is None:
			raise ValueError("Gr (hierarchical graph) is not defined and was not passed to Estaciones.__init__")
		if ubicaciones is None:
			raise ValueError("ubicaciones is not defined and was not passed to Estaciones.__init__")
		# Validation: ensure ubicaciones has keys for Gr nodes
		missing = [n for n in Gr if n not in ubicaciones]
		if missing:
			snippet = missing[:10]
			more = len(missing) - len(snippet)
			more_text = f", ...(+{more} more)" if more > 0 else ""
			raise KeyError(f"'ubicaciones' does not contain entries for Gr nodes. Missing keys: {snippet}{more_text}")
		t = []
		s = 1
		enr = {}
		for i in Gr:
			enr[i] = [i, float("inf")]
			estacion_c = EstacionCarga(10, [], [], random.choice(ubicaciones[i]), i, s, enr)
			t.append(estacion_c)
			s += 1
		self.total = t

	def __repr__(self):
		m = ""
		for i in self.total:
			m += repr(i) + "\n"
		return m


__all__ = [
	'Auto', 'Flota', 'Enrutador', 'Servidor', 'Antenas', 'EstacionCarga', 'Estaciones',
]


