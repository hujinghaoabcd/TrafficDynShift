import numpy as np

from trafficdynshift.data import TrafficWindowDataset, chronological_split


def test_window_dataset_shapes() -> None:
    data = np.arange(100 * 5, dtype=np.float32).reshape(100, 5)
    ds = TrafficWindowDataset(data, in_len=12, out_len=12)
    x, y = ds[0]
    assert x.shape == (12, 5, 1)
    assert y.shape == (12, 5)
    assert len(ds) == 77


def test_chronological_split() -> None:
    data = np.zeros((100, 3), dtype=np.float32)
    train, val, test = chronological_split(data, 0.6, 0.2)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
