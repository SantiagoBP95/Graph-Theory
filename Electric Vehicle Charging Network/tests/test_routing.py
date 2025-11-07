from ev_model.routing import dijkstra
import networkx as nx


def test_dijkstra_networkx():
    G = nx.path_graph(4)
    # add weights
    for u,v in G.edges():
        G[u][v]['weight'] = 1
    order, dist, prev = dijkstra(G, 0)
    assert dist[3] == 3
