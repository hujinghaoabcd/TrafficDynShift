# Data layout

Raw PeMS data are intentionally excluded from Git.

The default Hydra config expects:

```text
data/
├── PeMSD3/
│   ├── PeMSD3.npz
│   └── adjacency.npy
├── PeMSD4/
│   ├── PeMSD4.npz
│   └── adjacency.npy
├── PeMSD7/
│   ├── PeMSD7.npz
│   └── adjacency.npy
└── PeMSD8/
    ├── PeMSD8.npz
    └── adjacency.npy
```

Traffic `.npz` files may contain either `data` with shape `(T,N,F)` or a first array with shape `(T,N)` / `(T,N,F)`. `feature_index: 0` selects flow.

The current adjacency convention is `adjacency[target, source] > 0`. For symmetric PeMS distance graphs the distinction is immaterial. Directed graphs must be converted accordingly before training.
