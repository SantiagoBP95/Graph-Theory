import ev_model.entities as e


def test_entities_expose():
    assert hasattr(e, 'Flota')
    assert hasattr(e, 'Estaciones')


def test_flota_basic():
    import networkx as nx
    G = nx.Graph()
    G.add_node(1)
    f = e.Flota(3, G=G)
    assert len(f.total) == 3
