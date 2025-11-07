from ev_model.utils import fc


def test_fc_numeric():
    assert fc(0) == 0.0
    assert fc(1) == 1.0
    assert isinstance(fc(1), float)


def test_fc_bad():
    assert fc('nope') == 0.0
