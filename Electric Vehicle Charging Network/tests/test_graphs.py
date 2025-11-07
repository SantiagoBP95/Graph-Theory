from ev_model.graphs import build_G, build_Ga


def test_build_G():
    # Provide minimal nodes/edges explicitly (module no longer provides defaults)
    nodes = ['A', 'B', 'C']
    edges = [('A', 'B', 1.0), ('B', 'C', 2.0)]
    G = build_G(nodes=nodes, edges=edges)
    assert len(G.nodes()) == 3
    assert G.number_of_edges() == 2


def test_build_Ga_weights():
    nodes = ['X', 'Y']
    edges = [('X', 'Y')]
    Ga = build_Ga(nodes=nodes, edges=edges, default_weight=0.5)
    for u, v in Ga.edges():
        assert 'weight' in Ga[u][v]
