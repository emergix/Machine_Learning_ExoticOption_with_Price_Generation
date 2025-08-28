#!/usr/bin/env python3
"""
Train a customizable PyTorch model from scratch with sane defaults.

Features
- CSV dataset loader (regression or classification)
- Optional time-series windowing (seq_len)
- Built-in models: MLP, LSTM; or dynamically import your own
- Optimizers: Adam, AdamW, SGD; schedulers: cosine/step/none
- Mixed precision (AMP), gradient clipping, early stopping
- Checkpointing (best + last) & resume, deterministic seeding
- Minimal metric set: regression (MSE/MAE/RMSE), classification (accuracy)

Quickstart
----------
Regression (MLP) on CSV:
    python train_pytorch_customizable.py \
        --csv-path data.csv --target-col y \
        --model mlp --mlp-dims 128 64 \
        --loss mse --epochs 50 --batch-size 256

Classification (MLP):
    python train_pytorch_customizable.py \
        --csv-path data.csv --target-col label \
        --model mlp --mlp-dims 256 128 \
        --loss cross_entropy --epochs 30

Sequence (LSTM) for time-series forecasting:
    python train_pytorch_customizable.py \
        --csv-path data.csv --target-col y --seq-len 32 \
        --model lstm --lstm-hidden 128 --lstm-layers 2 --loss mse

Use your own model (dynamic import):
    python train_pytorch_customizable.py \
        --csv-path data.csv --target-col y \
        --custom-model-path my_model.py --custom-class MyNet \
        --custom-kwargs '{"hidden_sizes":[128,64],"dropout":0.1}'

Notes
- For LSTM with seq_len>1, the dataset builds sliding windows over features.
- If --feature-cols is omitted, all non-target numeric columns are used.
- For classification with cross_entropy, labels must be integer class IDs [0..C-1].
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# place ce helper en haut du fichier (après les imports)
import torch

def make_grad_scaler(device_str: str, enabled: bool):
    """
    Créé un GradScaler compatible:
    - PyTorch ≥ 2.2 : torch.amp.GradScaler('cuda'|'cpu'|'mps', enabled=...)
    - PyTorch 2.0–2.1 : torch.amp.GradScaler(...?) peut ne pas accepter device
    - PyTorch < 2.0 : torch.cuda.amp.GradScaler(enabled=...)
    """
    dev = "cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else (
          "mps"  if device_str.startswith("mps")  else "cpu")
    # 1) Essai API récente (positional arg)
    try:
        return torch.amp.GradScaler(dev, enabled=enabled)   # PyTorch ≥ 2.2
    except TypeError:
        pass
    # 2) Essai API récente (sans device)
    try:
        return torch.amp.GradScaler(enabled=enabled)        # PyTorch 2.0–2.1
    except Exception:
        pass
    # 3) Fallback API historique
    from torch.cuda.amp import GradScaler as OldGradScaler
    return OldGradScaler(enabled=enabled)


try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    pd = None

# -------------------- Utils --------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def to_device(batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device):
    x, y = batch
    return x.to(device), y.to(device)


# -------------------- Dataset --------------------

class CSVDataset(Dataset):
    """CSV dataset with optional time-series windowing (seq_len).

    If seq_len == 1 (default), each row -> (x, y).
    If seq_len > 1, constructs sliding windows over X to predict y at t (next-step or aligned).
    """

    def __init__(
        self,
        csv_path: str,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        seq_len: int = 1,
        pred_horizon: int = 0,
        dtype: str = "float32",
        dropna: bool = True,
    ):
        if pd is None:
            raise RuntimeError("pandas is required. Please `pip install pandas`.\n")
        df = pd.read_csv(csv_path)
        if dropna:
            df = df.dropna()
        assert target_col in df.columns, f"target_col '{target_col}' not in CSV"
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found. Specify --feature-cols.")

        X = df[feature_cols].to_numpy().astype(dtype)
        y = df[target_col].to_numpy().astype(dtype)

        self.seq_len = max(1, int(seq_len))
        self.pred_horizon = int(pred_horizon)

        if self.seq_len == 1:
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)
        else:
            # Build sliding windows X[t-seq_len+1:t+1] -> y[t + pred_horizon]
            T = len(X)
            last_idx = T - self.pred_horizon
            windows = []
            targets = []
            for t in range(self.seq_len - 1, last_idx):
                t_y = t + self.pred_horizon
                if t_y >= T:
                    break
                win = X[t - self.seq_len + 1 : t + 1]
                windows.append(win)
                targets.append(y[t_y])
            self.X = torch.tensor(np.stack(windows))  # [N, seq_len, F]
            self.y = torch.tensor(np.array(targets))   # [N]

        # Standardize features (optional simple z-score). For seq data, standardize per feature.
        # Comment out if you prefer raw scale.
        self._standardize_inplace()

    def _standardize_inplace(self):
        x = self.X
        if x.ndim == 2:  # [N, F]
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True).clamp_min(1e-8)
            self.X = (x - mean) / std
        elif x.ndim == 3:  # [N, L, F]
            mean = x.mean(dim=(0,1), keepdim=True)
            std = x.std(dim=(0,1), keepdim=True).clamp_min(1e-8)
            self.X = (x - mean) / std

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        if x.ndim == 1:
            x = x
        else:
            x = x  # [seq_len, F]
        return x, y


# -------------------- Models --------------------

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1, hidden: Iterable[int] = (128, 64), dropout: float = 0.0, bn: bool = True, activation: str = "relu"):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        act = acts.get(activation.lower(), nn.ReLU)
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:  # [B, L, F] -> pool over L
            x = x.mean(dim=1)
        return self.net(x)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.0, out_features: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        dir_mult = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_size * dir_mult, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:  # [B, F] -> [B, 1, F]
            x = x.unsqueeze(1)
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def build_model(args, input_size: int, num_classes_or_out: int) -> nn.Module:
    if args.custom_model_path:
        # Dynamically import user model
        sys.path.append(os.path.abspath(os.path.dirname(args.custom_model_path)))
        module_name = Path(args.custom_model_path).stem
        mod = __import__(module_name)
        cls = getattr(mod, args.custom_class)
        kwargs = json.loads(args.custom_kwargs) if args.custom_kwargs else {}
        model: nn.Module = cls(input_size=input_size, out_features=num_classes_or_out, **kwargs)
        return model

    if args.model == "mlp":
        return MLP(
            in_features=input_size,
            out_features=num_classes_or_out,
            hidden=args.mlp_dims,
            dropout=args.dropout,
            bn=not args.no_bn,
            activation=args.activation,
        )
    elif args.model == "lstm":
        return LSTMRegressor(
            input_size=input_size,
            hidden_size=args.lstm_hidden,
            num_layers=args.lstm_layers,
            bidirectional=args.lstm_bidir,
            dropout=args.dropout,
            out_features=num_classes_or_out,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


# -------------------- Training --------------------

@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = math.inf  # for regression (lower is better). For classification we'll invert.
    steps: int = 0


def make_optimizer(params, args):
    if args.opt == "adam":
        return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.opt == "adamw":
        return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.opt == "sgd":
        return optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    raise ValueError(args.opt)


def make_scheduler(optimizer, args):
    if args.sched == "none":
        return None
    if args.sched == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.sched == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    raise ValueError(args.sched)


def get_loss_fn(args):
    if args.loss == "mse":
        return nn.MSELoss()
    if args.loss == "mae":
        return nn.L1Loss()
    if args.loss == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if args.loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(args.loss)


def is_classification(args) -> bool:
    return args.loss in {"bce", "cross_entropy"}


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, classification: bool) -> Dict[str, float]:
    with torch.no_grad():
        if classification:
            if y_pred.ndim == 2 and y_pred.size(1) > 1:
                pred = y_pred.argmax(dim=1)
            else:
                pred = (torch.sigmoid(y_pred).view_as(y_true) > 0.5).long()
            acc = (pred == y_true.long()).float().mean().item()
            return {"acc": acc}
        else:
            mse = torch.mean((y_pred.squeeze() - y_true.squeeze()) ** 2).item()
            mae = torch.mean(torch.abs(y_pred.squeeze() - y_true.squeeze())).item()
            rmse = math.sqrt(mse)
            return {"mse": mse, "mae": mae, "rmse": rmse}


def save_checkpoint(out_dir: Path, model: nn.Module, optimizer, scheduler, state: TrainState, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "state": asdict(state),
    }, path)
    return path


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None) -> TrainState:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    st = TrainState(**ckpt.get("state", {}))
    return st


# -------------------- Main --------------------

class metaparameters(object):
    def __init__(self):
        self.INPUT_DIM=4
        self.INPUT_GOAL=0
        self.NB_NEURON_PRINCIPAL =20
        self.ACTIVATION_PRINCIPALE = 'relu'
        self.ACTIVATION_PRINCIPALE_FINALE = 'linear'
        self.ADDITIONAL_LEARNING_NB_EPOCH=1000
        self.NBLAYERS =11
        self.VERBOSE_FLAG=2
        self.NB_ADDITIONAL_LEARNING =0
        self.OPTIMIZER = 'adams'
        self.PATH=''
        self.LEARNINGBASE_ORIGIN=""
        self.LEARNINGBASE_BUT=""
        self.nbDate=12
        self.nbTraj=10000
        self.X_scaler = 0
        self.X_scaled = 0
        self.X = 0
        self.pkgPath = 0
        self.smallestNbDate = 4
        self.largestNbDate = 24
        self.smallestmaturity = 1
        self.largestmaturity = 5
        self.smallestF = 70
        self.largestF = 150
        self.largestmu = 0.05
        self.smallestmu = 0.0
        self.smallestSigma = 0.15
        self.largestSigma = 0.3
        self.smallestRho = 0.4
        self.largestRho = 1
        self.smallestBonus = -2
        self.largestBonus = 2
        self.smallestYetiBarrier = 90
        self.largestYetiBarrier = 110
        self.smallestYetiCoupon = 0
        self.largestYetiCoupon = 2
        self.smallestPhoenixBarrier = 80
        self.largestPhoenixBarrier = 100
        self.smallestPhoenixCoupon = 0.5
        self.largestPhoenixCoupon = 2
        self.smallestPDIBarrier = 40
        self.largestPDIBarrier = 70
        self.smallestPDIGearing = -5
        self.largestPDIGearing = +5
        self.smallestPDIStrike = 40
        self.largestPDIStrike = 70
        self.smallestPDIType = -1
        self.largestPDIType = 1

def build_dataloaders(args) -> Tuple[DataLoader, DataLoader, int, int]:
    # Build dataset
    ds = CSVDataset(
        csv_path=args.csv_path,
        target_col=args.target_col,
        feature_cols=args.feature_cols,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        dtype="float32",
        dropna=True,
    )

    n = len(ds)
    val_size = int(n * args.val_size)
    if args.timeseries_split:
        # Keep chronological order
        train_idx = list(range(0, n - val_size))
        val_idx = list(range(n - val_size, n))
    else:
        idx = list(range(n))
        random.shuffle(idx)
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]

    from torch.utils.data import Subset

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # Infer input and output sizes
    sample_x, sample_y = ds[0]
    if sample_x.ndim == 1:
        input_size = sample_x.shape[0]
    else:
        input_size = sample_x.shape[-1]

    if is_classification(args) and args.loss == "cross_entropy":
        num_out = args.num_classes
    else:
        num_out = 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=not args.timeseries_split, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, input_size, num_out


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, args) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        x, y = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
            logits = model(x)
            if loss_fn.__class__.__name__ == "CrossEntropyLoss":
                loss = loss_fn(logits, y.long())
            elif loss_fn.__class__.__name__ == "BCEWithLogitsLoss":
                loss = loss_fn(logits.view_as(y).float(), y.float())
            else:
                loss = loss_fn(logits.view_as(y), y)
        if args.amp:
            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / max(1, n_batches)

    # compute a quick metric on a small sample to avoid full pass (optional). Here we skip and return empty.
    return avg_loss, {}


def evaluate(model, loader, loss_fn, device, args) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_y = []
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            x, y = to_device(batch, device)
            logits = model(x)
            if loss_fn.__class__.__name__ == "CrossEntropyLoss":
                loss = loss_fn(logits, y.long())
            elif loss_fn.__class__.__name__ == "BCEWithLogitsLoss":
                loss = loss_fn(logits.view_as(y).float(), y.float())
            else:
                loss = loss_fn(logits.view_as(y), y)
            total_loss += loss.item()
            n_batches += 1
            all_y.append(y.detach())
            all_pred.append(logits.detach())
    avg_loss = total_loss / max(1, n_batches)
    y_true = torch.cat(all_y, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    metrics = compute_metrics(y_true, y_pred, classification=is_classification(args))
    return avg_loss, metrics




# -------------------- Notebook-friendly runner --------------------

@dataclass
class TrainConfig:
    # Data
    csv_path: str = ""
    target_col: str = ""
    feature_cols: Optional[List[str]] = None
    seq_len: int = 1
    pred_horizon: int = 0
    val_size: float = 0.2
    timeseries_split: bool = False
    num_classes: int = 2

    # Model
    model: str = "mlp"  # ["mlp", "lstm"]
    mlp_dims: List[int] = None
    activation: str = "relu"  # ["relu","gelu","tanh","silu"]
    lstm_hidden: int = 128
    lstm_layers: int = 1
    lstm_bidir: bool = False
    dropout: float = 0.0
    no_bn: bool = False

    # Custom external model
    custom_model_path: Optional[str] = None
    custom_class: Optional[str] = None
    custom_kwargs: Optional[str] = None  # JSON string

    # Training
    loss: str = "mse"  # ["mse","mae","huber","bce","cross_entropy"]
    opt: str = "adamw"  # ["adam","adamw","sgd"]
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 50
    grad_clip: Optional[float] = None
    amp: bool = False
    num_workers: int = 0

    # Scheduler
    sched: str = "none"  # ["none","step","cosine"]
    step_size: int = 10
    gamma: float = 0.5

    # Early stopping / checkpoints
    patience: int = 10
    out_dir: str = "outputs"
    resume: Optional[str] = None

    # Misc
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

     # Nouveau champ : affichage intermédiaire
    nb_intermediary_epoch: int = 1   # par défaut on logge chaque epoch


    def __post_init__(self):
        if self.mlp_dims is None:
            self.mlp_dims = [128, 64]


def run_training(cfg: TrainConfig) -> dict:
    """
    Lance un entraînement en utilisant une instance TrainConfig (idéal pour notebook).
    Retourne un petit dict de sortie (meilleure métrique, chemin des checkpoints, etc.)
    """
    # on réutilise toutes les fonctions existantes qui attendent "args"
    args = cfg  # alias: même API attributaire

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    train_loader, val_loader, input_size, out_features = build_dataloaders(args)
    model = build_model(args, input_size=input_size, num_classes_or_out=out_features).to(device)
    loss_fn = get_loss_fn(args)
    optimizer = make_optimizer(model.parameters(), args)
    scheduler = make_scheduler(optimizer, args)
    scaler = make_grad_scaler(device_str="cuda", enabled=args.amp)

    state = TrainState(epoch=0, best_metric=(1e9 if not is_classification(args) else -1e9))

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            state = load_checkpoint(ckpt_path, model, optimizer, scheduler)
            print(f"Resumed from {ckpt_path} at epoch {state.epoch} best_metric={state.best_metric}")
        else:
            print(f"Warning: resume checkpoint {ckpt_path} not found.")

    print(model)
    print(f"Training on {device} for {args.epochs} epochs.\n")

    no_improve = 0
    for epoch in range(state.epoch, args.epochs):
        state.epoch = epoch
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, args)
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device, args)

        if scheduler is not None:
            scheduler.step()

        metric_name = "acc" if is_classification(args) else "rmse"
        metric_value = val_metrics.get(metric_name, -val_loss if is_classification(args) else val_loss)

        improved = (metric_value < state.best_metric) if not is_classification(args) else (metric_value > state.best_metric)
        if improved:
            state.best_metric = metric_value
            save_checkpoint(out_dir, model, optimizer, scheduler, state, tag="best")
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(out_dir, model, optimizer, scheduler, state, tag="last")

        msg = {"epoch": epoch + 1, "train_loss": float(train_loss), "val_loss": float(val_loss)}
        msg.update({k: float(v) for k, v in val_metrics.items()})
        print(msg)

        if no_improve >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    print("Done. Best metric:", state.best_metric)
    print(f"Checkpoints saved to: {out_dir.resolve()} (best.pt, last.pt)")

    return {
        "best_metric": float(state.best_metric),
        "out_dir": str(out_dir.resolve()),
        "device": str(device),
        "input_size": int(input_size),
        "out_features": int(out_features),
        "model_class": model.__class__.__name__,
    }


# -------------------- CLI fallback --------------------

def _parse_cli_to_config(argv: Optional[List[str]] = None) -> TrainConfig:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # (mêmes arguments que ton main actuel)
    # Data
    p.add_argument("--csv-path", type=str, required=True)
    p.add_argument("--target-col", type=str, required=True)
    p.add_argument("--feature-cols", type=str, nargs="*", default=None)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--pred-horizon", type=int, default=0)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--timeseries-split", action="store_true")
    p.add_argument("--num-classes", type=int, default=2)
    # Model
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "lstm"])
    p.add_argument("--mlp-dims", type=int, nargs="*", default=[128, 64])
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "tanh", "silu"])
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--lstm-bidir", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-bn", action="store_true")
    # Custom
    p.add_argument("--custom-model-path", type=str, default=None)
    p.add_argument("--custom-class", type=str, default=None)
    p.add_argument("--custom-kwargs", type=str, default=None)
    # Training
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber", "bce", "cross_entropy"])
    p.add_argument("--opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    # Scheduler
    p.add_argument("--sched", type=str, default="none", choices=["none", "step", "cosine"])
    p.add_argument("--step-size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.5)
    # Early stopping / checkpoints
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--resume", type=str, default=None)
    # Misc
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    a = p.parse_args(argv)
    return TrainConfig(
        csv_path=a.csv_path,
        target_col=a.target_col,
        feature_cols=a.feature_cols,
        seq_len=a.seq_len,
        pred_horizon=a.pred_horizon,
        val_size=a.val_size,
        timeseries_split=a.timeseries_split,
        num_classes=a.num_classes,
        model=a.model,
        mlp_dims=a.mlp_dims,
        activation=a.activation,
        lstm_hidden=a.lstm_hidden,
        lstm_layers=a.lstm_layers,
        lstm_bidir=a.lstm_bidir,
        dropout=a.dropout,
        no_bn=a.no_bn,
        custom_model_path=a.custom_model_path,
        custom_class=a.custom_class,
        custom_kwargs=a.custom_kwargs,
        loss=a.loss,
        opt=a.opt,
        lr=a.lr,
        weight_decay=a.weight_decay,
        batch_size=a.batch_size,
        epochs=a.epochs,
        grad_clip=a.grad_clip,
        amp=a.amp,
        num_workers=a.num_workers,
        sched=a.sched,
        step_size=a.step_size,
        gamma=a.gamma,
        patience=a.patience,
        out_dir=a.out_dir,
        resume=a.resume,
        seed=a.seed,
        device=a.device,
    )


from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import time
import torch

# Utilise tqdm auto (bon rendu sous notebook et console)
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # si tqdm n'est pas installé, on continue sans barre

def _infer_shapes_from_loader(loader: torch.utils.data.DataLoader) -> Tuple[int, int]:
    xb, yb = next(iter(loader))
    if xb.ndim == 2:
        input_size = xb.shape[1]
    elif xb.ndim == 3:
        input_size = xb.shape[-1]
    else:
        raise ValueError(f"Unexpected input ndim={xb.ndim}")
    out_features = 1 if yb.ndim <= 1 else yb.shape[1]
    return int(input_size), int(out_features)

def run_training_with_loaders(
    cfg,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    input_size: Optional[int] = None,
    out_features: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Entraîne en utilisant des DataLoader fournis (aucune lecture CSV).
    Ajouts:
      - Barre de progression par époque (train), optionnelle pour val
      - Temps par époque et temps total
      - Affichage 'tous les n epochs' via cfg.nb_intermediary_epoch (défaut=1)

    Options (facultatives) lues dans cfg:
      - nb_intermediary_epoch: int (défaut=1)
      - show_progress: bool (défaut=True)
      - show_progress_val: bool (défaut=False)  -> barre aussi sur la boucle validation
      - progress_leave: bool (défaut=False)     -> laisse la barre affichée
    """
    args = cfg
    set_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # options d’affichage
    nb_intermediary_epoch = getattr(args, "nb_intermediary_epoch", 1)
    show_progress = getattr(args, "show_progress", True)
    show_progress_val = getattr(args, "show_progress_val", False)
    progress_leave = getattr(args, "progress_leave", False)

    # Inférer shapes si besoin
    if input_size is None or out_features is None:
        inf_in, inf_out = _infer_shapes_from_loader(train_loader)
        input_size = input_size or inf_in
        out_features = out_features or inf_out

    # Modèle / optim / loss / sched
    model = build_model(args, input_size=input_size, num_classes_or_out=out_features).to(device)
    loss_fn = get_loss_fn(args)
    optimizer = make_optimizer(model.parameters(), args)
    scheduler = make_scheduler(optimizer, args)
    scaler = make_grad_scaler(args.device, enabled=(args.amp and "cuda" in args.device and torch.cuda.is_available()))
    classification = is_classification(args)

    state = TrainState(epoch=0, best_metric=(-1e9 if classification else 1e9))
    history = {"epoch": [], "train_loss": [], "val_loss": [], "metric": [], "epoch_time_sec": []}

    # Reprise éventuelle
    if getattr(args, "resume", None):
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            state = load_checkpoint(ckpt_path, model, optimizer, scheduler)
            print(f"Resumed from {ckpt_path} at epoch {state.epoch} best_metric={state.best_metric}")
        else:
            print(f"Warning: resume checkpoint {ckpt_path} not found.")

    # Infos device
    print(model)
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}  |  batches/train: {len(train_loader)}  val: {len(val_loader)}")
    print(f"Training on {device} for {args.epochs} epochs.\n")

    no_improve = 0
    t0_total = time.perf_counter()

    for epoch in range(state.epoch, args.epochs):
        state.epoch = epoch
        t0 = time.perf_counter()

        # ---------------- Train ----------------
        model.train()
        total_loss, n_batches = 0.0, 0

        iterator = train_loader
        if show_progress and tqdm is not None:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=progress_leave)

        for x, y in iterator:
            x = x.to(device, dtype=torch.float32)
            if classification and args.loss == "cross_entropy":
                y = y.to(device, dtype=torch.long)
            else:
                y = y.to(device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(args.amp and device.type=="cuda")):
                logits = model(x)
                if loss_fn.__class__.__name__ == "CrossEntropyLoss":
                    loss = loss_fn(logits, y.long())
                elif loss_fn.__class__.__name__ == "BCEWithLogitsLoss":
                    loss = loss_fn(logits.view_as(y).float(), y.float())
                else:
                    loss = loss_fn(logits.view_as(y), y)

            if args.amp and device.type == "cuda":
                scaler.scale(loss).backward()
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            total_loss += float(loss.item()); n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        # ---------------- Val ----------------
        model.eval()
        total_loss, n_batches = 0.0, 0
        all_y, all_pred = [], []

        val_iter = val_loader
        if show_progress_val and tqdm is not None:
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=progress_leave)

        with torch.no_grad():
            for x, y in val_iter:
                x = x.to(device, dtype=torch.float32)
                y_in = y
                if classification and args.loss == "cross_entropy":
                    y = y.to(device, dtype=torch.long)
                else:
                    y = y.to(device, dtype=torch.float32)

                logits = model(x)
                if loss_fn.__class__.__name__ == "CrossEntropyLoss":
                    loss = loss_fn(logits, y.long())
                elif loss_fn.__class__.__name__ == "BCEWithLogitsLoss":
                    loss = loss_fn(logits.view_as(y).float(), y.float())
                else:
                    loss = loss_fn(logits.view_as(y), y)

                total_loss += float(loss.item()); n_batches += 1
                all_y.append(y_in.detach().cpu()); all_pred.append(logits.detach().cpu())

        val_loss = total_loss / max(1, n_batches)
        y_true = torch.cat(all_y, 0)
        y_pred = torch.cat(all_pred, 0)
        val_metrics = compute_metrics(y_true, y_pred, classification=classification)

        if scheduler is not None:
            scheduler.step()

        # Métrique pour early stopping
        metric_name = "acc" if classification else "rmse"
        metric_value = val_metrics.get(metric_name, -val_loss if classification else val_loss)
        improved = (metric_value > state.best_metric) if classification else (metric_value < state.best_metric)

        if improved:
            state.best_metric = metric_value
            save_checkpoint(out_dir, model, optimizer, scheduler, state, tag="best")
            no_improve = 0
        else:
            no_improve += 1
        save_checkpoint(out_dir, model, optimizer, scheduler, state, tag="last")

        # Historique + temps d’époque
        epoch_time = time.perf_counter() - t0
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["metric"].append(float(metric_value))
        history["epoch_time_sec"].append(float(epoch_time))

        # Affichage (tous les n epochs + toujours 1er/dernier)
        if ((epoch + 1) % nb_intermediary_epoch == 0) or epoch == 0 or epoch == args.epochs - 1:
            msg = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "epoch_time_sec": round(epoch_time, 2),
            }
            msg.update({k: float(v) for k, v in val_metrics.items()})
            print(msg)

        if no_improve >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    total_time = time.perf_counter() - t0_total
    # petit formatage lisible
    hh = int(total_time // 3600)
    mm = int((total_time % 3600) // 60)
    ss = int(total_time % 60)

    print(f"Done. Best metric: {state.best_metric}")
    print(f"Total training time: {hh:02d}:{mm:02d}:{ss:02d}  ({total_time:.2f}s)")
    print(f"Checkpoints saved to: {out_dir.resolve()} (best.pt, last.pt)")
    if getattr(cfg, "export_bundle", True):
        feature_stats = compute_feature_stats_from_loader(train_loader)
        save_training_bundle(
            out_dir=out_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            cfg=cfg,
            input_size=input_size,
            out_features=out_features,
            feature_stats=feature_stats,
            tag="best"
        )
    return {
        "best_metric": float(state.best_metric),
        "out_dir": str(out_dir.resolve()),
        "device": str(device),
        "input_size": int(input_size),
        "out_features": int(out_features),
        "model_class": model.__class__.__name__,
        "history": history,
        "total_time_sec": float(total_time),
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "state": state,
    }



import torch
import numpy as np

import torch

def notation(y_true: torch.Tensor, y_pred: torch.Tensor, rmse_train: float | None = None) -> float:
    """
    Calcule une note unique (0–100) qui mesure la performance de calibration en régression.
    Combine précision (NRMSE), corrélation, pente/intercept de calibration, et généralisation.

    Args:
        y_true: torch.Tensor des vraies valeurs (shape [N] ou [N,1])
        y_pred: torch.Tensor des prédictions (shape identique)
        rmse_train: float optionnel, RMSE du set d'entraînement
                    (sert à pénaliser l'overfitting)

    Returns:
        note: float entre 0 et 100 (100 = calibration parfaite)
    """
    y_true = y_true.detach().cpu().view(-1).float()
    y_pred = y_pred.detach().cpu().view(-1).float()

    # Échelle (std des vraies valeurs)
    scale = float(y_true.std().clamp_min(1e-8))

    # Erreurs
    diff = y_pred - y_true
    rmse = float(torch.sqrt(torch.mean(diff**2)))
    bias = float(diff.mean())
    nrmse = rmse / scale

    # Corrélation de Pearson
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    num = float((yt * yp).sum())
    den = float(torch.sqrt((yt**2).sum()) * torch.sqrt((yp**2).sum()))
    r = num / den if den > 0 else 0.0
    r = max(0.0, min(1.0, r))  # borné à [0,1]

    # Calibration linéaire (pente/intercept)
    X = torch.stack([y_true, torch.ones_like(y_true)], dim=1)  # [N,2]
    coeffs, *_ = torch.linalg.lstsq(X, y_pred.unsqueeze(1))
    slope, intercept = float(coeffs[0,0]), float(coeffs[1,0])
    slope_err = abs(slope - 1.0)
    intercept_norm = abs(intercept) / scale

    # Gap généralisation
    gap = max(0.0, (rmse - rmse_train) / scale) if rmse_train is not None else 0.0

    # Scores partiels (tous bornés entre 0 et 1, plus grand = mieux)
    q_precision   = 1.0 / (1.0 + nrmse)
    q_corr        = r
    q_calibration = 1.0 / (1.0 + slope_err + intercept_norm)
    q_general     = 1.0 / (1.0 + gap)

    # Pondérations (somme=1)
    weights = dict(precision=0.4, corr=0.2, calibration=0.3, general=0.1)

    score = (
        weights["precision"]   * q_precision +
        weights["corr"]        * q_corr +
        weights["calibration"] * q_calibration +
        weights["general"]     * q_general
    )
    return 100.0 * score

def rightScale(vect):
    a=min(vect)
    b=max(vect)
    a=floor(a)
    b=-floor(-b)
    return (a,b)

class PricingInterface:
    def __init__(self,updatefunc,contupdate,ShowBarsFlag,model):  
        self.ShowBarsFlag = ShowBarsFlag
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.vega1 = 0
        self.vega2 = 0
        self.vega3 = 0
        self.smin = -1
        self.smax = 1
        self.vmin = -1
        self.vmax = 1
        self.cmin = -1
        self.cmax = 1
        self.updateFuncIni=updatefunc
        self.ContinuousUpdating = contupdate
        self.ws1=widgets.FloatSlider(
            value=100,
            min=90,
            max=110.0,
            step=1,
            description='S1',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.1f'
            )
        self.ws2=widgets.FloatSlider(
            value=100,
            min=90,
            max=110.0,
            step=1,
            description='S2',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.ws3=widgets.FloatSlider(
            value=100,
            min=90,
            max=110.0,
            step=1,
            description='S3',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )    
        self.wmu1=widgets.FloatSlider(
            value=0.01,
            min=0,
            max=0.05,
            step=0.001,
            description='mu1',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wmu2=widgets.FloatSlider(
            value=0.01,
            min=0,
            max=0.05,
            step=0.001,
            description='mu2',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wmu3=widgets.FloatSlider(
            value=0.01,
            min=0,
            max=0.05,
            step=0.001,
            description='mu3',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wv1=widgets.FloatSlider(
            value=0.2,
            min=0.15,
            max=0.3,
            step=0.001,
            description='v1',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wv2=widgets.FloatSlider(
            value=0.2,
            min=0.15,
            max=0.3,
            step=0.001,
            description='v2',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wv3=widgets.FloatSlider(
            value=0.2,
            min=0.15,
            max=0.3,
            step=0.001,
            description='v3',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wc12=widgets.FloatSlider(
            value=0.8,
            min=0.4,
            max=1,
            step=0.001,
            description='c12',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wc13=widgets.FloatSlider(
            value=0.8,
            min=0.4,
            max=1,
            step=0.001,
            description='c13',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wc23=widgets.FloatSlider(
            value=0.8,
            min=0.4,
            max=1,
            step=0.001,
            description='c23',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wBonus=widgets.FloatSlider(
            value=0,
            min=-2,
            max=2,
            step=0.1,
            description='Bonus',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wYetiBarrier=widgets.FloatSlider(
            value=100,
            min=90,
            max=110,
            step=1,
            description='YetiBarrier',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wYetiCoupon=widgets.FloatSlider(
            value=1,
            min=0,
            max=2,
            step=0.1,
            description='YetiCoupon',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPhoenixBarrier=widgets.FloatSlider(
            value=90,
            min=80,
            max=100,
            step=1,
            description='PhoenixBarrier',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPhoenixCoupon=widgets.FloatSlider(
            value=1,
            min=0.5,
            max=2,
            step=0.1,
            description='PhoenixCoupon',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPDIBarrier=widgets.FloatSlider(
            value=60,
            min=40,
            max=70,
            step=1,
            description='PDIBarrier',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPDIStrike=widgets.FloatSlider(
            value=60,
            min=40,
            max=70,
            step=1,
            description='PDIStrike',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )

        self.wPDIGearing=widgets.FloatSlider(
            value=-1,
            min=-5,
            max=5,
            step=1,
            description='PDIGearing',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPDIType=widgets.FloatSlider(
            value=-1,
            min=-3,
            max=3,
            step=1,
            description='PDIType',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wMaturity=widgets.FloatSlider(
            value=3,
            min=1,
            max=7,
            step=0.1,
            description='Maturity',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wNbDate=widgets.IntSlider(
            value=12,
            min=4,
            max=24,
            step=1,
            description='NbDate',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='d'
            )
        self.wNbBoost=widgets.IntSlider(
            value=0,
            min=0,
            max=2,
            step=1,
            description='NbBoost',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='d'
            )
        self.model = model
       
        self.res1 = 0
        self.result = 0
        
        self.wtotal=widgets.VBox([widgets.HBox([self.ws1,self.ws2,self.ws3,self.wmu1,self.wmu2,self.wmu3,self.wv1,self.wv2,self.wv3,self.wc12,self.wc13,self.wc23]),\
                     widgets.HBox([self.wBonus,self.wYetiBarrier,self.wYetiCoupon,self.wPhoenixBarrier,self.wPhoenixCoupon,self.wPDIBarrier,\
                     self.wPDIStrike,self.wPDIGearing,self.wPDIType,self.wMaturity,self.wNbDate,self.wNbBoost])])      

        self.out = widgets.interactive_output(self.updateFunc1,\
            {'s1':self.ws1,'s2':self.ws2,'s3':self.ws3,'mu1':self.wmu1,'mu2':self.wmu2,'mu3':self.wmu3,'v1':self.wv1,'v2':self.wv2,'v3':self.wv3,\
            'c12':self.wc12,'c13':self.wc13,'c23':self.wc23,\
            'Bonus':self.wBonus,'YetiBarrier':self.wYetiBarrier,'YetiCoupon':self.wYetiCoupon,'PhoenixBarrier':self.wPhoenixBarrier,\
            'PhoenixCoupon':self.wPhoenixCoupon,'PDIBarrier':self.wPDIBarrier,'PDIGearing':self.wPDIGearing,\
            'PDIStrike':self.wPDIStrike,'PDIType':self.wPDIType,\
            'Maturity':self.wMaturity,'NbDate':self.wNbDate,'NbBoost':self.wNbBoost})
        sstep=(self.smax-self.smin)/100
        vstep=(self.vmax-self.vmin)/100

        self.fw1 = FloatProgress(value=self.result ,min=-10, max=10.0, step=0.1, description='price:',bar_style='info',orientation='vertical')
        self.fw2 = FloatProgress(value=self.delta1 ,min=self.smin, max=self.smax, step=sstep, description='delta1:',bar_style='info',orientation='vertical')
        self.fw3 = FloatProgress(value=self.delta2 ,min=self.smin, max=self.smax, step=sstep, description='delta2:',bar_style='info',orientation='vertical')
        self.fw4 = FloatProgress(value=self.delta3 ,min=self.smin, max=self.smax, step=sstep, description='delta3:',bar_style='info',orientation='vertical')
        self.fw5 = FloatProgress(value=self.vega1 ,min=self.vmin, max=self.vmax, step=vstep, description='vega1:',bar_style='info',orientation='vertical')
        self.fw6 = FloatProgress(value=self.vega2 ,min=self.vmin, max=self.vmax, step=vstep, description='vega2:',bar_style='info',orientation='vertical')
        self.fw7 = FloatProgress(value=self.vega3 ,min=self.vmin, max=self.vmax, step=vstep, description='vega3:',bar_style='info',orientation='vertical')
        
        if self.ShowBarsFlag:
            self.Wfinal = widgets.VBox([self.wtotal, self.out, widgets.HBox([self.fw1,self.fw2,self.fw3,self.fw4,self.fw5,self.fw6,self.fw7])])
        else :
            self.Wfinal = widgets.VBox([self.wtotal, self.out])
        

    def updateFunc1(self,s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,\
                     Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier,PhoenixCoupon,PDIBarrier,\
                     PDIStrike,PDIGearing,PDIType,Maturity,NbDate,NbBoost):
        
        a,smin,smax,vmin,vmax,cmin,cmax=self.updateFuncIni(\
                     s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,\
                     Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier,PhoenixCoupon,PDIBarrier,\
                     PDIStrike,PDIGearing,PDIType,Maturity,NbDate,NbBoost,self.model)
        
        self.result = a[0]
        self.delta1 = 100*(a[1]-a[0])/(s1/100)
        self.delta2 = 100*(a[2]-a[0])/(s2/100)
        self.delta3 = 100*(a[3]-a[0])/(s3/100)
        self.vega1 = 100*(a[4]-a[0])
        self.vega2 = 100*(a[5]-a[0])
        self.vega3 = 100*(a[6]-a[0])
       
        self.smin = smin
        self.smax = smax
        self.vmin = vmin
        self.vmax = vmax
        self.cmin = cmin
        self.cmax = cmax

        if self.ShowBarsFlag:
            print([self.result ,self.delta1,self.delta2,self.delta3,self.vega1,self.vega2,self.vega3])
            print(a)
            print([smin,smax,vmin,vmax,cmin,cmax])
            
    def display(self):
            display(self.Wfinal)
            
            
def recompute_interface(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier,\
                PhoenixCoupon,PDIBarrier,PDIGearing,PDIStrike,PDIType,Maturity,NbDate,NbBoost,model):  
    
    d0 = np.array([s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,\
                      Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier, PhoenixCoupon,PDIBarrier,\
                      PDIGearing, PDIStrike, PDIType,Maturity,NbDate]) 
    dx = np.zeros([7,23]) 
    for ip in range(7):
        dx[ip]=d0
    dx[1,0]*=1.01;dx[2,0]*=1.01;dx[3,0]*=1.01;dx[4,0]*=1.01;dx[5,0]*=1.01;dx[6,0]*=1.01
    a=predict_Option_new(dx,model)
    #a=predict_Option(dx,learnedModels,NbBoost)
    result=a[0];
    ligne_string = "Present Price = {}  Delta S1 = {}   Delta S2 = {}  Delta S3 = {}"
    ligne_string2 = "Vega S1 = {}   Vega S2 = {}  Vega S3 = {}"
    delta1=(a[1]-result)/(s1/100);delta2=(a[2]-result)/(s2/100);delta3=(a[3]-result)/(s3/100)
    vega1 = 100*(a[4]-result);vega2 = 100*(a[5]-result);vega3 = 100*(a[6]-result);
    nbpoints=10
    nbpoints2=20
    x = np.linspace(0.9, 1.1, num=nbpoints,endpoint=True)
    y = np.zeros(nbpoints)  
    z = np.zeros(nbpoints) 
    s = np.zeros([nbpoints,nbpoints]) 
    
    ds = np.zeros([nbpoints,23]) 
    dv = np.zeros([nbpoints,23]) 
    dc = np.zeros([nbpoints,23]) 
    for ip in range(nbpoints):
        ds[ip]= d0;dv[ip]= d0;dc[ip]= d0;
        ds[ip,0]*=x[ip];ds[ip,1]*=x[ip];ds[ip,2]*=x[ip]
        dv[ip,6]*=x[ip];dv[ip,7]*=x[ip];dv[ip,8]*=x[ip]
        dc[ip,9]=min(1,dc[ip,9]*x[ip]);dc[ip,10]=min(1,dc[ip,10]*x[ip]);dc[ip,11]=min(1,dc[ip,11]*x[ip])
    y=predict_Option_new(ds,model)
    #y=predict_Option(ds,learnedModels,NbBoost)
    z=predict_Option_new(dv,model)
    #z=predict_Option(dv,learnedModels,NbBoost)
    c=predict_Option_new(dc,model)
    #c=predict_Option(dc,learnedModels,NbBoost)
    print(ligne_string.format("%.4f" % result,"%.4f" % delta1,"%.4f" % delta2,"%.4f" % delta3)) 
    print(ligne_string2.format("%.4f" % vega1,"%.4f" % vega2,"%.4f" % vega3)) 
    xnew  = np.linspace(0.9, 1.1, num=nbpoints2,endpoint=True)
    y1 = interp1d(x,y,kind='cubic')
    z1 = interp1d(x,z,kind='cubic')
    c1 = interp1d(x,c,kind='cubic')
    fig, axs = pyplot.subplots(1, 3, figsize=(15, 5))
    displaymultiplier1 = 2
    displaymultiplier2 = 0.33
    ynew=y1(xnew)
    axs[0].plot(xnew,ynew)
    axs[0].set_title('Spot Ladder')
    rs=rightScale(ynew/displaymultiplier1)
    axs[0].set_ylim(rs[0]*displaymultiplier1,rs[1]*displaymultiplier1)
    znew=z1(xnew)
    axs[1].plot(xnew,znew)
    axs[1].set_title('Vega Ladder')
    zs=rightScale(znew/displaymultiplier2)
    axs[1].set_ylim(zs[0]*displaymultiplier2,zs[1]*displaymultiplier2)
    cnew=c1(xnew)
    axs[2].plot(xnew,cnew)
    axs[2].set_title('Cega Ladder')
    cs=rightScale(cnew/displaymultiplier2)
    axs[2].set_ylim(cs[0]*displaymultiplier2,cs[1]*displaymultiplier2)
    pyplot.show()
    return [a,-3,3,-3,3,-1,1]
