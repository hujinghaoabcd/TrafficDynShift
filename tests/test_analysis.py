import numpy as np

from trafficdynshift.analysis import jensen_shannon, morphology_descriptors


def test_similarity_helpers() -> None:
    assert jensen_shannon(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    a = np.array([[0, 1], [1, 0]], dtype=float)
    desc = morphology_descriptors(a)
    assert desc["num_nodes"] == 2.0
    assert desc["mean_degree"] == 1.0
