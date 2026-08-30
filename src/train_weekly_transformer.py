"""Train a compact Transformer on weekly user histories.

The feature source is still the base competition parquet: weekly tensors are
read from the leakage-safe folds produced by ``build_segmented_features.py``.
This script writes log-space predictions for fold_02, fold_03 and fold_end;
``blend_segmented_transformer.py`` combines them with the classical hierarchy.

PyTorch is optional for the classical pipeline.  Install it with::

    python -m pip install -r requirements_transformer.txt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler


FEATURES_DIR = Path("data/segmented_base")
PREDICTIONS_DIR = Path("data/segmented_predictions/transformer")
REPORT_PATH = Path("reports/weekly_transformer.json")

N_WEEKS = 26
SEQUENCE_CHANNELS = ["gmv", "to_ord", "searches", "to_cart"]
SEQUENCE_COLS = [
    f"week_{channel}_{week:02d}"
    for week in range(N_WEEKS)
    for channel in SEQUENCE_CHANNELS
]
STATIC_COLS = [
    "adi_26w",
    "cv2_demand_26w",
    "demand_weeks_26w",
    "zero_week_share_26w",
    "positive_week_gmv_mean_26w",
    "weekly_gmv_mean_26w",
    "weekly_gmv_max_26w",
    "gmv_sum_7d",
    "gmv_sum_30d",
    "gmv_sum_90d",
    "gmv_sum_180d",
    "searches_sum_7d",
    "searches_sum_30d",
    "searches_sum_90d",
    "to_ord_sum_7d",
    "to_ord_sum_30d",
    "to_ord_sum_90d",
    "to_cart_sum_30d",
    "recency_gmv_days",
    "recency_order_days",
    "recency_search_days",
    "recency_cart_days",
    "tenure_days",
    "gmv_lifetime",
    "orders_lifetime",
    "gmv_recent_share_7_30",
    "gmv_recent_share_30_90",
    "search_recent_share_7_30",
    "conv_order_per_search_90d",
    "conv_cart_per_search_90d",
    "conv_order_per_cart_90d",
    "gmv_per_order_90d",
]

STAGES = [
    ("fold_02", ["fold_00", "fold_01"]),
    ("fold_03", ["fold_00", "fold_01", "fold_02"]),
    ("fold_end", ["fold_00", "fold_01", "fold_02", "fold_03"]),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train weekly Transformer prediction branch.")
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Two epochs and a smaller network for an infrastructure check.",
    )
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 32:
        ap.error("--epochs must be positive and --batch-size >= 32")
    if args.d_model % args.heads:
        ap.error("--d-model must be divisible by --heads")
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.d_model = min(args.d_model, 32)
        args.layers = min(args.layers, 1)
        args.heads = min(args.heads, 4)
    return args


def import_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for the Transformer branch. Install "
            "requirements_transformer.txt or run only the classical pipeline."
        ) from exc
    return torch, nn, (DataLoader, Dataset)


def choose_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but CUDA is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("--device mps requested, but MPS is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def required_columns() -> list[str]:
    return ["user_id", "target", "demand_class_id", *SEQUENCE_COLS, *STATIC_COLS]


def read_fold(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Rebuild features with src/build_segmented_features.py."
        )
    schema = pl.read_parquet_schema(path)
    missing = sorted(set(required_columns()) - set(schema))
    if missing:
        raise ValueError(
            f"{path} lacks Transformer features ({missing[:8]}). "
            "Re-run feature builder with --overwrite."
        )
    return pl.read_parquet(path, columns=required_columns())


def sequence_array(df: pl.DataFrame) -> np.ndarray:
    # Files store week_00=newest; the Transformer receives oldest -> newest.
    values = np.empty((df.height, N_WEEKS, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    for out_pos, week in enumerate(reversed(range(N_WEEKS))):
        cols = [f"week_{channel}_{week:02d}" for channel in SEQUENCE_CHANNELS]
        values[:, out_pos, :] = df.select(cols).to_numpy().astype(np.float32, copy=False)
    return np.log1p(np.clip(values, 0.0, 1e9))


def static_array(df: pl.DataFrame) -> np.ndarray:
    values = df.select(STATIC_COLS).to_numpy().astype(np.float32, copy=False)
    return np.log1p(np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1e9))


def fit_normalizers(
    sequences: list[np.ndarray], statics: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    channel_sum = np.zeros(len(SEQUENCE_CHANNELS), dtype=np.float64)
    channel_sq = np.zeros(len(SEQUENCE_CHANNELS), dtype=np.float64)
    count = 0
    for seq in sequences:
        channel_sum += seq.sum(axis=(0, 1), dtype=np.float64)
        channel_sq += np.square(seq, dtype=np.float64).sum(axis=(0, 1))
        count += seq.shape[0] * seq.shape[1]
    mean = channel_sum / count
    var = np.maximum(channel_sq / count - mean**2, 1e-6)
    std = np.sqrt(var)
    scaler = StandardScaler().fit(np.concatenate(statics, axis=0))
    return mean.astype(np.float32), std.astype(np.float32), scaler


def normalize_sequence(seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((seq - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def build_model(torch: Any, nn: Any, args: argparse.Namespace, n_static: int) -> Any:
    class WeeklyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(len(SEQUENCE_CHANNELS), args.d_model)
            self.position = nn.Parameter(torch.zeros(1, N_WEEKS, args.d_model))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.heads,
                dim_feedforward=args.d_model * 4,
                dropout=args.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=args.layers)
            self.static = nn.Sequential(
                nn.Linear(n_static, 48),
                nn.GELU(),
                nn.LayerNorm(48),
            )
            self.class_embedding = nn.Embedding(4, 8)
            self.head = nn.Sequential(
                nn.Linear(args.d_model * 2 + 48 + 8, 96),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(96, 1),
            )

        def forward(self, seq: Any, static: Any, demand_class: Any) -> Any:
            encoded = self.encoder(self.input_projection(seq) + self.position)
            pooled = torch.cat([encoded[:, -1], encoded.mean(dim=1)], dim=1)
            combined = torch.cat(
                [pooled, self.static(static), self.class_embedding(demand_class)], dim=1
            )
            return self.head(combined).squeeze(1)

    return WeeklyTransformer()


def make_dataset_class(torch: Any, Dataset: Any) -> Any:
    class ArrayDataset(Dataset):
        def __init__(
            self,
            sequence: np.ndarray,
            static: np.ndarray,
            demand_class: np.ndarray,
            target: np.ndarray | None,
        ) -> None:
            self.sequence = sequence
            self.static = static
            self.demand_class = demand_class
            self.target = target

        def __len__(self) -> int:
            return len(self.sequence)

        def __getitem__(self, index: int) -> tuple[Any, ...]:
            items: tuple[Any, ...] = (
                torch.from_numpy(self.sequence[index]),
                torch.from_numpy(self.static[index]),
                torch.tensor(int(self.demand_class[index]), dtype=torch.long),
            )
            if self.target is not None:
                items += (torch.tensor(float(self.target[index]), dtype=torch.float32),)
            return items

    return ArrayDataset


def train_and_predict_stage(
    torch: Any,
    nn: Any,
    DataLoader: Any,
    Dataset: Any,
    device: Any,
    args: argparse.Namespace,
    train_folds: list[str],
    eval_fold: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.time()
    print(f"\n=== Transformer {train_folds} -> {eval_fold} ===", flush=True)
    train_frames = [read_fold(args.features_dir / f"{fold}.parquet") for fold in train_folds]
    eval_df = read_fold(args.features_dir / f"{eval_fold}.parquet")

    train_sequences = [sequence_array(df) for df in train_frames]
    train_statics = [static_array(df) for df in train_frames]
    mean, std, static_scaler = fit_normalizers(train_sequences, train_statics)
    seq_train = np.concatenate(
        [normalize_sequence(seq, mean, std) for seq in train_sequences], axis=0
    )
    static_train = np.concatenate(
        [static_scaler.transform(x).astype(np.float32) for x in train_statics], axis=0
    )
    class_train = np.concatenate(
        [df["demand_class_id"].to_numpy().astype(np.int64) for df in train_frames]
    )
    y_train = np.concatenate(
        [np.log1p(np.clip(df["target"].to_numpy(), 0.0, None)) for df in train_frames]
    ).astype(np.float32)

    seq_eval = normalize_sequence(sequence_array(eval_df), mean, std)
    static_eval = static_scaler.transform(static_array(eval_df)).astype(np.float32)
    class_eval = eval_df["demand_class_id"].to_numpy().astype(np.int64)
    y_eval = (
        np.clip(eval_df["target"].to_numpy(), 0.0, None).astype(np.float64)
        if eval_df["target"].null_count() == 0
        else np.full(eval_df.height, np.nan, dtype=np.float64)
    )

    train_ds = Dataset(seq_train, static_train, class_train, y_train)
    eval_ds = Dataset(seq_eval, static_eval, class_eval, None)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    model = build_model(torch, nn, args, len(STATIC_COLS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    criterion = nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    losses: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for seq, static, demand_class, target in train_loader:
            seq = seq.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            demand_class = demand_class.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                pred = model(seq, static, demand_class)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(seq)
            seen += len(seq)
        scheduler.step()
        epoch_loss = loss_sum / max(seen, 1)
        losses.append(epoch_loss)
        print(
            f"  epoch {epoch + 1:02d}/{args.epochs}: "
            f"train_rmse={math.sqrt(epoch_loss):.6f}",
            flush=True,
        )

    model.eval()
    pred_parts: list[np.ndarray] = []
    with torch.no_grad():
        for seq, static, demand_class in eval_loader:
            pred = model(
                seq.to(device, non_blocking=True),
                static.to(device, non_blocking=True),
                demand_class.to(device, non_blocking=True),
            )
            pred_parts.append(pred.detach().cpu().numpy())
    z_pred = np.clip(np.concatenate(pred_parts), 0.0, 20.0).astype(np.float32)

    diagnostics: dict[str, Any] = {
        "train_folds": train_folds,
        "eval_fold": eval_fold,
        "train_rows": len(seq_train),
        "eval_rows": len(seq_eval),
        "epoch_train_rmse": [round(math.sqrt(x), 6) for x in losses],
        "seconds": round(time.time() - started, 1),
    }
    if np.isfinite(y_eval).all():
        z_true = np.log1p(y_eval)
        diagnostics["rmsle"] = round(float(np.sqrt(np.mean((z_true - z_pred) ** 2))), 6)
        print(f"  {eval_fold} RMSLE={diagnostics['rmsle']:.6f}", flush=True)

    users = eval_df["user_id"].to_numpy().astype(np.int64)
    del (
        train_frames,
        eval_df,
        train_sequences,
        train_statics,
        seq_train,
        static_train,
        class_train,
        y_train,
        seq_eval,
        static_eval,
        train_ds,
        eval_ds,
        train_loader,
        eval_loader,
        model,
        optimizer,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return z_pred, users, class_eval, diagnostics


def main() -> None:
    args = parse_args()
    torch, nn, loader_types = import_torch()
    DataLoader, Dataset = loader_types
    set_seed(torch, args.seed)
    device = choose_device(torch, args.device)
    print(f"PyTorch {torch.__version__}; device={device}", flush=True)

    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    stage_reports: dict[str, Any] = {}
    for eval_fold, train_folds in STAGES:
        pred, users, classes, diagnostics = train_and_predict_stage(
            torch,
            nn,
            DataLoader,
            make_dataset_class(torch, Dataset),
            device,
            args,
            train_folds,
            eval_fold,
        )
        out = args.predictions_dir / f"{eval_fold}.parquet"
        pl.DataFrame(
            {
                "user_id": users,
                "demand_class_id": classes.astype(np.int8),
                "z_transformer": pred,
            }
        ).write_parquet(out, compression="zstd", statistics=True)
        print(f"  saved {out}", flush=True)
        stage_reports[eval_fold] = diagnostics

    report = {
        "source": "weekly sequences derived directly from data/train.parquet",
        "sequence": {"weeks": N_WEEKS, "channels": SEQUENCE_CHANNELS},
        "static_features": STATIC_COLS,
        "parameters": {
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "seed": args.seed,
            "quick": args.quick,
        },
        "stages": stage_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.predictions_dir / "metadata.json").write_text(
        json.dumps(report["parameters"], indent=2, ensure_ascii=False)
    )
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
