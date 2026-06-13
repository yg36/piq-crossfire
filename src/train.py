from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ARTIFACT_DIR, MODEL_PATH, QUALITY_LABELS, STATUS_LABELS, SUMMARY_PATH
from .model import PIQCrossfireNet, set_seed
from .synthetic_data import generate_incident_dataset
from .text_features import TextFeatureExtractor


def _split_indices(size: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(size)
    test_size = int(size * test_fraction)
    return indices[test_size:], indices[:test_size]


def _quality_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    prediction = probabilities >= 0.45
    target_bool = target >= 0.5
    tp = (prediction & target_bool).sum().item()
    fp = (prediction & ~target_bool).sum().item()
    fn = (~prediction & target_bool).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    exact = (prediction == target_bool).all(dim=1).float().mean().item()
    return {
        "quality_precision": float(precision),
        "quality_recall": float(recall),
        "quality_f1": float(f1),
        "quality_exact_match": float(exact),
    }


def _status_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == target).float().mean().item())


def _evaluate(
    model: PIQCrossfireNet,
    x: torch.Tensor,
    y_quality: torch.Tensor,
    y_strength: torch.Tensor,
    y_status: torch.Tensor,
    quality_loss: nn.Module,
    strength_loss: nn.Module,
    status_loss: nn.Module,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        quality_logits, strength, status_logits = model(x)
        loss = (
            quality_loss(quality_logits, y_quality)
            + 0.55 * strength_loss(strength, y_strength)
            + 0.35 * status_loss(status_logits, y_status)
        )
        metrics = _quality_metrics(quality_logits, y_quality)
        metrics.update(
            {
                "loss": float(loss.item()),
                "strength_mae": float(torch.abs((strength - y_strength) * 100).mean().item()),
                "status_accuracy": _status_accuracy(status_logits, y_status),
            }
        )
        return metrics


def train_crossfire_model(
    *,
    n_samples: int = 8500,
    epochs: int = 90,
    batch_size: int = 128,
    learning_rate: float = 0.0025,
    seed: int = 42,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, Any]:
    set_seed(seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_incident_dataset(n_samples=n_samples, seed=seed)
    extractor = TextFeatureExtractor()
    x_np = extractor.transform(dataset["texts"])  # type: ignore[arg-type]
    y_quality_np = dataset["labels"]  # type: ignore[assignment]
    y_strength_np = dataset["strengths"]  # type: ignore[assignment]
    y_status_np = dataset["statuses"]  # type: ignore[assignment]

    train_idx, test_idx = _split_indices(len(x_np), test_fraction=0.20, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.tensor(x_np[train_idx], dtype=torch.float32),
        torch.tensor(y_quality_np[train_idx], dtype=torch.float32),
        torch.tensor(y_strength_np[train_idx], dtype=torch.float32),
        torch.tensor(y_status_np[train_idx], dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x_test = torch.tensor(x_np[test_idx], dtype=torch.float32, device=device)
    y_quality_test = torch.tensor(y_quality_np[test_idx], dtype=torch.float32, device=device)
    y_strength_test = torch.tensor(y_strength_np[test_idx], dtype=torch.float32, device=device)
    y_status_test = torch.tensor(y_status_np[test_idx], dtype=torch.long, device=device)

    model = PIQCrossfireNet(input_dim=extractor.output_dim, quality_count=len(QUALITY_LABELS)).to(device)
    quality_loss = nn.BCEWithLogitsLoss()
    strength_loss = nn.SmoothL1Loss()
    status_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.012)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    stale_epochs = 0
    patience = 18

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for x_batch, y_quality_batch, y_strength_batch, y_status_batch in train_loader:
            x_batch = x_batch.to(device)
            y_quality_batch = y_quality_batch.to(device)
            y_strength_batch = y_strength_batch.to(device)
            y_status_batch = y_status_batch.to(device)

            quality_logits, strength, status_logits = model(x_batch)
            loss = (
                quality_loss(quality_logits, y_quality_batch)
                + 0.55 * strength_loss(strength, y_strength_batch)
                + 0.35 * status_loss(status_logits, y_status_batch)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        scheduler.step()
        metrics = _evaluate(model, x_test, y_quality_test, y_strength_test, y_status_test, quality_loss, strength_loss, status_loss)
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = float(np.mean(batch_losses))
        history.append(metrics)

        if metrics["loss"] < best_loss - 1e-4:
            best_loss = metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    final_metrics = _evaluate(model, x_test, y_quality_test, y_strength_test, y_status_test, quality_loss, strength_loss, status_loss)
    final_metrics = {key: round(value, 4) for key, value in final_metrics.items()}

    model_path = artifact_dir / MODEL_PATH.name
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "model_config": {
                "input_dim": extractor.output_dim,
                "quality_count": len(QUALITY_LABELS),
                "hidden_dim": 160,
                "status_count": len(STATUS_LABELS),
            },
            "quality_labels": QUALITY_LABELS,
            "status_labels": STATUS_LABELS,
            "hash_dim": extractor.hash_dim,
            "metrics": final_metrics,
            "history": history,
            "seed": seed,
            "n_samples": n_samples,
        },
        model_path,
    )

    summary = {
        "model_path": str(model_path),
        "metrics": final_metrics,
        "epochs_completed": len(history),
        "n_samples": n_samples,
        "device_used": str(device),
    }
    (artifact_dir / SUMMARY_PATH.name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

