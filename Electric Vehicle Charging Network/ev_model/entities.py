import random
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

	def AOC(self, n, E, G=None):
		"""Search for a charging-station route using randomized walks.

		This method performs `n` random walk trials starting from the vehicle's
		current location. For each walk it accumulates a heuristic score based
		on inverse edge weight (favoring short/cheap edges). It returns the
		route (list of nodes) with the highest score. If the vehicle is
		already located at a station, the current location is returned.

		Parameters
		- n: number of trials
		- E: Estaciones-like container with `.total` entries that have an
			 `ubicacion` attribute
		- G: optional NetworkX graph to use for neighbors/weights. If not
			 provided, the function attempts to obtain a global `G` from the
			 notebook namespace via a helper `_get_global` (used historically).

		Returns
		- list of node labels representing the chosen route, or the current
		  ubicacion if no route is found.
		"""
		G = G or _get_global('G')
		if G is None:
			raise ValueError("G (global city graph) is not defined and was not passed to Auto.AOC")

		Caminos = []
		CaminosM = []
		spc = []
		# extract station locations from the Estaciones container
		E_nodos = [x.ubicacion for x in E.total]
		if self.ubicacion in E_nodos:
			# already at a station: return a route list for consistency
			return [self.ubicacion]
		else:
			for i in range(n):
				Ch = [self.ubicacion]
				eCh = 0
				P = []
				# perform a random walk, avoiding repeated nodes
				while Ch[-1] not in E_nodos:
					vecinos = [a for a in G.neighbors(Ch[-1]) if a not in Ch]
					if len(vecinos) > 0:
						j = random.choice(vecinos)
						dab = G[Ch[-1]][j].get('weight', 1)
						Ch.append(j)
						# collect inverse-weight heuristic (guard divide-by-zero)
						P.append(1 / dab if dab != 0 else 0)
						eCh += 1
					else:
						# no new neighbors: truncated path
						CaminosM.append(Ch)
						break
				Caminos.append(Ch)
				# compute trial score safely (avoid division by zero)
				sp = (sum(P) * (100 / eCh)) if eCh != 0 else 0
				spc.append(sp)
			if len(spc) == 0:
				return self.ubicacion
			indice = spc.index(max(spc))
			return Caminos[indice]

	def route_dijkstra(self, E, G=None, measure_time: bool = False):
		"""Find a route to the closest charging station using Dijkstra.

		This method uses the package's `dijkstra` dispatcher where possible and
		falls back to NetworkX shortest-path reconstruction if predecessor
		pointers are not provided by the dispatcher for NetworkX graphs.

		Parameters
		- E: Estaciones-like container with `.total` entries that have `ubicacion`
			 attributes
		- G: optional NetworkX graph; if not provided, attempts to obtain global `G`
		- measure_time: if True returns a tuple (route, elapsed_seconds)

		Returns
		- route: list of node labels representing the shortest path to the
		  nearest station (by path length). If measure_time True, returns
		  (route, elapsed_seconds).
		"""
		G = G or _get_global('G')
		if G is None:
			raise ValueError("G (global city graph) is not defined and was not passed to route_dijkstra")

		E_nodos = [x.ubicacion for x in E.total]
		if self.ubicacion in E_nodos:
			# already at a station: return a route list; when measuring time, return (route_list, 0.0)
			result = [self.ubicacion]
			return (result, 0.0) if measure_time else result

		start_time = time.perf_counter() if measure_time else None

		# Use the routing.dijkstra dispatcher to get distances; for NetworkX
		# graphs the returned 'prev' may be empty, so we handle that case.
		order, dist, prev = dijkstra(G, self.ubicacion)

		# find nearest station by distance mapping
		try:
			# dist may be a dict-like mapping of distances
			nearest = min((s for s in E_nodos if s in dist), key=lambda s: dist.get(s, float('inf')))
		except ValueError:
			# no reachable stations
			result = [self.ubicacion]
			if measure_time:
				return result, 0.0
			return result

		# reconstruct path: if prev is available use it, otherwise use NetworkX
		if prev and prev.get(nearest) is not None:
			# reconstruct via predecessor pointers
			path = []
			cur = nearest
			while cur is not None:
				path.append(cur)
				cur = prev.get(cur)
			path = list(reversed(path))
		else:
			# fallback: use NetworkX shortest path (guaranteed if graph connected)
			try:
				path = nx.shortest_path(G, source=self.ubicacion, target=nearest, weight='weight')
			except Exception:
				# if shortest_path fails, return current location
				path = [self.ubicacion]

		if measure_time:
			elapsed = time.perf_counter() - start_time
			return path, elapsed
		return path


class Flota:
	"""Fleet container that builds `total` Auto instances.

	Constructor will create `total` vehicles and place them at random nodes
	on the provided graph `G`. If `G` is not provided the global `G` is
	attempted via `_get_global` (historical notebook compatibility).
	"""

	def __init__(self, total, G=None):
		# if no G is provided, fall back to global notebook variable
		G = G or _get_global('G')
		if G is None:
			raise ValueError("G (global city graph) is not defined and was not passed to Flota.__init__")
		f = []
		for i in range(total):
			placa = "car" + str(i)
			bateria = random.randrange(50, 100)
			ubicacion = random.choice(list(G.nodes))
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
		Gr = Gr or _get_global('Gr')
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
		Ga = Ga or _get_global('Ga')
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
		Gr = Gr or _get_global('Gr')
		ubicaciones = ubicaciones or _get_global('ubicaciones')
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


