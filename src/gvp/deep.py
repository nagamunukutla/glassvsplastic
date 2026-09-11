"""Optional deep-learning tier (PyTorch + timm). Guarded: the repo runs without torch.

Install the deep extra first::

    pip install -r requirements-deep.txt

Everything here is written so that *the same manifests* feed both tiers, which is what makes
the classical-vs-deep comparison in ``docs/06_model_comparison.md`` methodologically honest:
identical splits, identical labels, identical test-time degradations.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import synth

# --------------------------------------------------------------------------------------
# optional imports
# --------------------------------------------------------------------------------------


class DeepUnavailable(RuntimeError):
    """Raised with install instructions when torch/timm is missing."""


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise DeepUnavailable(
            "PyTorch is not installed in this environment. The deep tier is optional; the "
            "classical tier and the whole reporting layer run without it.\n"
            "Install with:  pip install -r requirements-deep.txt"
        ) from exc
    return __import__("torch"), __import__("torchvision")


def torch_available() -> bool:
    try:
        _require_torch()
        return True
    except DeepUnavailable:
        return False


# A curated list of backbones worth benchmarking for glass-vs-plastic. Names are timm names
# where possible; ``tv:`` prefix means torchvision. See docs/06_model_comparison.md for the
# rationale (parameters, FLOPs, evidence on waste datasets, glass/plastic-specific notes).
RECOMMENDED_BACKBONES: Tuple[str, ...] = (
    "tv:resnet18",              # classical reference / sanity baseline
    "tv:resnet50",
    "tv:densenet121",           # repeatedly the best performer on TrashNet in the literature
    "tv:efficientnet_b0",
    "tv:mobilenet_v3_large",    # edge deployment default
    "tv:mobilenet_v3_small",
    "tv:convnext_tiny",         # best all-round modern CNN in independent backbone studies
    "tv:swin_tiny_patch4_window7_224",
    "tv:vit_b_16",              # needs more data/aug than CNNs when fine-tuned
    "mobilenetv4_conv_small.e2400_r224_in1k",
    "efficientnet_b0.ra4_e3600_r224_in1k",
)


@dataclass
class TrainConfig:
    model: str = "tv:resnet18"
    data_dir: str = "data/processed/proxy"
    out_dir: str = "results/deep"
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    img_size: int = 224
    freeze_backbone: bool = False
    label_smoothing: float = 0.05
    mixup: float = 0.0
    num_workers: int = 2
    seed: int = 0
    train_split: str = "train"
    val_split: str = "val"
    test_splits: Sequence[str] = ("test",)
    device: str = "auto"
    amp: bool = True
    class_balance: bool = True
    channels_last: bool = True
    notes: str = ""
    extra: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def _build_dataset(data_dir: str, split: str, img_size: int, train: bool):
    torch, _tv = _require_torch()
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    rows = synth.load_manifest(os.path.join(data_dir, f"manifest_{split}.csv"))
    if train:
        tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.12)),
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15),
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.08)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    class _ManifestDataset(Dataset):
        def __init__(self, root: str, rows: List[dict], tf):
            self.root, self.rows, self.tf = root, rows, tf

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            r = self.rows[i]
            img = Image.open(os.path.join(self.root, r["path"])).convert("RGB")
            return self.tf(img), int(r["label"])

    return _ManifestDataset(data_dir, rows, tf), rows


# --------------------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------------------


def build_backbone(name: str, num_classes: int = 2, pretrained: bool = True, freeze: bool = False):
    torch, tv = _require_torch()
    if name.startswith("tv:"):
        spec = name[3:]
        if not hasattr(tv.models, spec):
            raise ValueError(f"torchvision has no model {spec!r}")
        try:
            weights = "DEFAULT" if pretrained else None
            model = getattr(tv.models, spec)(weights=weights)
        except Exception as exc:  # offline / no cached weights
            print(f"[deep] could not load pretrained weights ({exc}); using random init")
            model = getattr(tv.models, spec)(weights=None)
        if hasattr(model, "head"):  # efficientnet / convnext / swin
            in_f = model.head.in_features if hasattr(model.head, "in_features") else model.head[-1].in_features
            model.head = torch.nn.Linear(in_f, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, torch.nn.Sequential):
                in_f = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_f, num_classes)
            else:  # resnet / densenet
                in_f = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_f, num_classes)
        elif hasattr(model, "fc"):
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        try:
            import timm
        except Exception as exc:  # pragma: no cover
            raise DeepUnavailable("timm not installed; `pip install -r requirements-deep.txt`") from exc
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    if freeze:
        for pname, p in model.named_parameters():
            if not any(k in pname for k in ("head", "fc", "classifier")):
                p.requires_grad = False
    return model


# --------------------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------------------


def train_and_eval(cfg: TrainConfig) -> Dict[str, object]:
    """Fine-tune one backbone and evaluate on the in-distribution + shifted test splits."""
    torch, _tv = _require_torch()
    from torch.utils.data import DataLoader

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None)
                                                           and torch.backends.mps.is_available() else "cpu")
    print(f"[deep] model={cfg.model} device={device} epochs={cfg.epochs} img={cfg.img_size}")

    tr_ds, tr_rows = _build_dataset(cfg.data_dir, cfg.train_split, cfg.img_size, train=True)
    va_ds, _ = _build_dataset(cfg.data_dir, cfg.val_split, cfg.img_size, train=False)
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=(device == "cuda"), drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_backbone(cfg.model, 2, pretrained=True, freeze=cfg.freeze_backbone).to(device)
    if cfg.channels_last and device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    y = np.array([int(r["label"]) for r in tr_rows])
    if cfg.class_balance:
        counts = np.bincount(y, minlength=2).astype(float)
        w = (counts.sum() / (2 * np.maximum(counts, 1)))
        class_w = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        class_w = None

    crit = torch.nn.CrossEntropyLoss(weight=class_w, label_smoothing=cfg.label_smoothing)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(1, cfg.epochs * max(1, len(tr_ld))), pct_start=0.25)
    try:  # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device == "cuda"))

    history: List[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        t0, losses = time.perf_counter(), []
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            if cfg.channels_last and device == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=scaler.is_enabled()):
                out = model(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            losses.append(float(loss.detach()))
        vm = _evaluate(model, va_ld, device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_balanced_accuracy": vm["balanced_accuracy"],
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"[deep] epoch {epoch + 1:>3}/{cfg.epochs} loss={np.mean(losses):.4f} "
              f"val_BA={vm['balanced_accuracy']:.4f} ({history[-1]['seconds']}s)")

    # final evaluation on every requested split
    from . import evaluate
    os.makedirs(cfg.out_dir, exist_ok=True)
    rows: List[dict] = []
    for split in cfg.test_splits:
        if not os.path.exists(os.path.join(cfg.data_dir, f"manifest_{split}.csv")):
            continue
        ds, _ = _build_dataset(cfg.data_dir, split, cfg.img_size, train=False)
        ld = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        pred, score, true, ms = _evaluate(model, ld, device, return_predictions=True)
        m = evaluate.binary_metrics(true, pred, score)
        m.update({"model": cfg.model, "split": split, "epochs": cfg.epochs,
                  "img_size": cfg.img_size, "params_m": round(count_params(model) / 1e6, 2),
                  "macs_g": round(count_macs(model, cfg.img_size) / 1e9, 2),
                  "latency_ms_per_image": round(ms, 3), "device": device,
                  "freeze_backbone": cfg.freeze_backbone, "notes": cfg.notes})
        rows.append(m)

    with open(os.path.join(cfg.out_dir, "deep_results.csv"), "a", newline="") as fh:
        cols = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=cols)
        if fh.tell() == 0:
            w.writeheader()
        w.writerows(rows)
    with open(os.path.join(cfg.out_dir, f"history_{cfg.model.replace('/', '_')}.json"), "w") as fh:
        json.dump({"config": {k: v for k, v in cfg.__dict__.items() if k != "extra"},
                   "history": history}, fh, indent=2, default=str)
    return {"rows": rows, "history": history}


def _evaluate(model, loader, device: str, return_predictions: bool = False):
    torch, _tv = _require_torch()
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    model.eval()
    preds, scores, trues = [], [], []
    t0 = None
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if t0 is None:
                _ = model(xb)  # warm-up before timing
                t0 = time.perf_counter()
            out = model(xb)
            p = torch.softmax(out.float(), dim=1)[:, 1]
            preds.append((p > 0.5).long().cpu().numpy())
            scores.append(p.cpu().numpy())
            trues.append(yb.numpy())
            n += len(yb)
    dt = (time.perf_counter() - t0) / max(n, 1) * 1000.0 if t0 else float("nan")
    pred = np.concatenate(preds) if preds else np.array([])
    score = np.concatenate(scores) if scores else np.array([])
    true = np.concatenate(trues) if trues else np.array([])
    if return_predictions:
        return pred, score, true, dt
    cm = confusion_matrix(true, pred, labels=[0, 1]) if len(true) else np.zeros((2, 2))
    return {"balanced_accuracy": float(balanced_accuracy_score(true, pred)) if len(true) else float("nan"),
            "accuracy": float((pred == true).mean()) if len(true) else float("nan"),
            "confusion": cm.tolist(), "latency_ms_per_image": dt}


# --------------------------------------------------------------------------------------
# cost accounting without extra dependencies
# --------------------------------------------------------------------------------------


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_macs(model, img_size: int = 224, device: str = "cpu") -> int:
    """Approximate MACs via forward hooks on Conv2d/Linear (no fvcore/thop dependency)."""
    torch, _tv = _require_torch()
    macs = {"total": 0}
    handles = []

    def conv_hook(mod, inp, out):
        try:
            k = mod.kernel_size[0] * mod.kernel_size[1]
            macs["total"] += int(out.shape[1] * out.shape[2] * out.shape[3] * k * mod.in_channels / mod.groups)
        except Exception:
            pass

    def lin_hook(mod, inp, out):
        macs["total"] += int(mod.in_features * mod.out_features)

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, img_size, img_size, device=device))
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    return macs["total"]


def export_onnx(model_name: str, out_path: str, img_size: int = 224, num_classes: int = 2) -> str:
    """Export a fine-tuned/torchvision backbone to ONNX for edge deployment."""
    torch, _tv = _require_torch()
    model = build_backbone(model_name, num_classes, pretrained=True).eval()
    dummy = torch.zeros(1, 3, img_size, img_size)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.onnx.export(model, dummy, out_path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    return out_path


def benchmark_latency(model_name: str, img_size: int = 224, repeats: int = 30, device: str = "auto") -> Dict[str, float]:
    """Latency + size + compute for the deployment discussion (docs/10_deployment.md)."""
    torch, _tv = _require_torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_backbone(model_name, 2, pretrained=True).to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return {"model": model_name, "device": device, "img_size": img_size,
            "latency_ms_median": float(np.median(times)),
            "latency_ms_p95": float(np.percentile(times, 95)),
            "params_m": count_params(model) / 1e6,
            "macs_g": count_macs(model, img_size, "cpu") / 1e9,
            "fps_batch1": 1000.0 / float(np.median(times))}
