import pytest

from netsim import DelayLine, NetSim


def test_clean_network_is_instant():
    n = NetSim()
    assert all(n.decide() == 0.0 for _ in range(100))


def test_loss_rate_matches():
    n = NetSim(loss=0.1, seed=1)
    drops = sum(n.decide() is None for _ in range(10_000))
    assert 900 < drops < 1100


def test_delay_stays_within_jitter_range():
    n = NetSim(delay_ms=100, jitter_ms=30, seed=2)
    ds = [n.decide() for _ in range(2000)]
    assert all(0.070 <= d <= 0.130 for d in ds)
    assert max(ds) - min(ds) > 0.05  # it actually varies


def test_delay_never_negative():
    n = NetSim(delay_ms=5, jitter_ms=50, seed=3)
    assert all(n.decide() >= 0 for _ in range(1000))


def test_same_seed_same_network():
    a, b = NetSim(10, 5, 0.2, seed=4), NetSim(10, 5, 0.2, seed=4)
    assert [a.decide() for _ in range(200)] == [b.decide() for _ in range(200)]


@pytest.mark.parametrize("kw", [{"loss": 1.5}, {"loss": -0.1}, {"delay_ms": -1}, {"jitter_ms": -1}])
def test_invalid_settings(kw):
    with pytest.raises(ValueError):
        NetSim(**kw)


def test_delay_line_sends_in_due_order():
    out = []
    line = DelayLine(out.append)
    line.put(0.05, b"a")
    line.put(0.0, b"b")
    line.put(0.02, b"c")
    line.close()  # waits until everything has been sent
    assert out == [b"b", b"c", b"a"]
