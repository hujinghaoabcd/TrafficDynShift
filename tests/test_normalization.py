import torch

from trafficdynshift.utils.normalization import denormalize_forecast, normalize_forecast_pair


def test_causal_pair_round_trip() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 12, 5, 1) * 3 + 10
    y = torch.randn(4, 12, 5) * 2 + 11
    _, yn, mean, std = normalize_forecast_pair(x, y)
    restored = denormalize_forecast(yn, mean, std)
    assert torch.allclose(restored, y, atol=1e-5)
