from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from trafficdynshift import ModelConfig, PDRForecaster
from trafficdynshift.data import load_region, make_loader
from trafficdynshift.graph import k_hop_edge_index
from trafficdynshift.training import evaluate, mean_region_mae, save_checkpoint, train_one_region
from trafficdynshift.utils import resolve_device, seed_everything


def build_model_config(cfg: DictConfig) -> ModelConfig:
    model = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model, dict)
    model.pop("name", None)
    model["in_len"] = int(cfg.data.in_len)
    model["out_len"] = int(cfg.data.out_len)
    return ModelConfig(**model)


def load_named_region(cfg: DictConfig, name: str):
    spec = cfg.data.regions[name]
    return load_region(
        name=name,
        data_path=spec.data_path,
        adjacency_path=spec.adjacency_path,
        in_len=int(cfg.data.in_len),
        out_len=int(cfg.data.out_len),
        feature_index=int(cfg.data.feature_index),
        train_ratio=float(cfg.data.train_ratio),
        val_ratio=float(cfg.data.val_ratio),
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.runtime.seed), bool(cfg.runtime.deterministic))
    device = resolve_device(str(cfg.runtime.device))
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)

    source_names = [str(x) for x in cfg.experiment.source_regions]
    target_name = str(cfg.experiment.target_region)
    if bool(cfg.experiment.strict_zero_shot) and target_name in source_names:
        raise ValueError("strict_zero_shot requires target_region to be absent from source_regions")

    sources = {name: load_named_region(cfg, name) for name in source_names}
    target = load_named_region(cfg, target_name)
    model = PDRForecaster(build_model_config(cfg)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.training.epochs),
        eta_min=float(cfg.training.min_lr),
    )

    source_edges = {
        name: k_hop_edge_index(
            region.adjacency,
            hops=int(cfg.data.support_hops),
            threshold=float(cfg.data.adjacency_threshold),
            include_self=bool(cfg.data.include_self),
        )
        for name, region in sources.items()
    }
    target_edges = k_hop_edge_index(
        target.adjacency,
        hops=int(cfg.data.support_hops),
        threshold=float(cfg.data.adjacency_threshold),
        include_self=bool(cfg.data.include_self),
    )

    source_train_loaders = {
        name: make_loader(
            region.train,
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
        )
        for name, region in sources.items()
    }
    source_val_loaders = {
        name: make_loader(
            region.val,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
        )
        for name, region in sources.items()
    }
    target_test_loader = make_loader(
        target.test,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
    )

    best_metric = float("inf")
    bad_epochs = 0
    best_path = output_dir / "best.pt"
    history: list[dict[str, object]] = []

    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_losses = {}
        for name in source_names:
            train_losses[name] = train_one_region(
                model,
                source_train_loaders[name],
                source_edges[name],
                optimizer,
                device,
                grad_clip=float(cfg.training.grad_clip),
            )

        val_results = [
            evaluate(model, source_val_loaders[name], source_edges[name], device)
            for name in source_names
        ]
        val_metric = mean_region_mae(val_results)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses,
                "source_val_mae": val_metric,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        print(f"epoch={epoch:03d} source_val_mae={val_metric:.4f}")

        if val_metric < best_metric:
            best_metric = val_metric
            bad_epochs = 0
            save_checkpoint(model, best_path, epoch, val_metric)
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.training.patience):
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    target_result = evaluate(model, target_test_loader, target_edges, device)
    summary = {
        "target_region": target_name,
        "source_regions": source_names,
        "seed": int(cfg.runtime.seed),
        "best_source_val_mae": best_metric,
        "target_test": target_result.__dict__,
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
