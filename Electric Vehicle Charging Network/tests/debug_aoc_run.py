import sys
import os
import networkx as nx
# ensure package path: add the package folder (current directory) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ev_model.entities import Auto, Estaciones

# Build a simple city graph G
G = nx.Graph()
G.add_edge('A', 'B', weight=1)
G.add_edge('B', 'C', weight=1)
G.add_edge('C', 'D', weight=1)

# Case 1: estación en 'A', auto ubicado en 'A'
Ga = ['g1']
ubicaciones = {'g1': ('A',)}
estaciones = Estaciones(Ga, ubicaciones)
print('Estaciones ubicaciones:', [e.ubicacion for e in estaciones.total])
auto1 = Auto('car1', 80, 'A')
print('Auto1 ubicacion:', auto1.ubicacion)
print('AOC result (start at station):', auto1.AOC(10, estaciones, G=G))

# Case 2: estación en 'C', auto at 'A' so path should reach station C
ubicaciones2 = {'g1': ('C',)}
estaciones2 = Estaciones(Ga, ubicaciones2)
auto2 = Auto('car2', 80, 'A')
print('\nEstaciones2 ubicaciones:', [e.ubicacion for e in estaciones2.total])
print('Auto2 ubicacion:', auto2.ubicacion)
print('AOC result (path to station):', auto2.AOC(50, estaciones2, G=G))

# Case 3: estación mapping with different type (int) to check type mismatch
G2 = nx.Graph()
G2.add_edge(1, 2, weight=1)
G2.add_edge(2, 3, weight=1)
Ga2 = ['g1']
ubicaciones3 = {'g1': (1,)}
estaciones3 = Estaciones(Ga2, ubicaciones3)
auto3 = Auto('car3', 80, 1)
print('\nEstaciones3 ubicaciones:', [e.ubicacion for e in estaciones3.total])
print('Auto3 ubicacion:', auto3.ubicacion)
print('AOC result (int nodes):', auto3.AOC(10, estaciones3, G=G2))

# Case 4: station node present but represented differently (string with spaces)
G3 = nx.Graph()
G3.add_edge(' X', 'Y', weight=1)
G3.add_edge('Y', 'Z', weight=1)
Ga3 = ['g1']
ubicaciones4 = {'g1': ('X',)}  # note 'X' vs ' X'
estaciones4 = Estaciones(Ga3, ubicaciones4)
auto4 = Auto('car4', 80, ' X')
print('\nEstaciones4 ubicaciones:', [e.ubicacion for e in estaciones4.total])
print('Auto4 ubicacion:', auto4.ubicacion)
try:
    print('AOC result (mismatched whitespace):', auto4.AOC(10, estaciones4, G=G3))
except Exception as e:
    print('Error running AOC for case 4:', e)
