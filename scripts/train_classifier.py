"""Train the EfficientNet-B0 glaucoma vs non-glaucoma classifier.

    python scripts/train_classifier.py --epochs 25 --batch 24
"""
import argparse
import time

import _path  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.datasets import ClsDataset
from src.metrics import classification_metrics
from src.models import build_classifier
from src.utils import plot_history, set_seed


def run_epoch(model, loader, loss_fn, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot, n, probs, labels = 0.0, 0, [], []
    use_amp = device == "cuda"
    for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", enabled=use_amp):
            logit = model(x).squeeze(1)
            loss = loss_fn(logit.float(), y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        tot += loss.item() * x.size(0)
        n += x.size(0)
        probs.append(torch.sigmoid(logit.detach().float()).cpu().numpy())
        labels.append(y.cpu().numpy())
    return tot / max(n, 1), np.concatenate(labels), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.CLS_EPOCHS)
    ap.add_argument("--batch", type=int, default=config.CLS_BATCH)
    ap.add_argument("--lr", type=float, default=config.CLS_LR)
    ap.add_argument("--size", type=int, default=config.CLS_IMG_SIZE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--splits", default=str(config.SPLITS_DIR))
    ap.add_argument("--out", default=str(config.CLS_CKPT))
    args = ap.parse_args()

    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    tr = ClsDataset(f"{args.splits}/cls_train.csv", args.size, train=True)
    va = ClsDataset(f"{args.splits}/cls_val.csv", args.size, train=False)
    pos = int(tr.labels.sum())
    neg = len(tr) - pos
    print(f"train={len(tr)} (glaucoma={pos}, non={neg}) val={len(va)}")
    kw = dict(num_workers=args.workers, pin_memory=device == "cuda")
    tl = DataLoader(tr, args.batch, shuffle=True, drop_last=len(tr) > args.batch, **kw)
    vl = DataLoader(va, args.batch, shuffle=False, **kw)

    model = build_classifier(pretrained=not args.no_pretrained).to(device)
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = max(1, min(2, args.epochs // 5))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - warm), eta_min=args.lr / 50),
    ], milestones=[warm])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best, bad, hist = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trl, ytr, ptr = run_epoch(model, tl, loss_fn, device, opt, scaler)
        vll, yva, pva = run_epoch(model, vl, loss_fn, device)
        sched.step()
        m = classification_metrics(yva, pva, 0.5)
        auc = m["roc_auc"] if not np.isnan(m["roc_auc"]) else m["accuracy"]
        hist.append(dict(epoch=ep, train_loss=trl, val_loss=vll, val_acc=m["accuracy"], val_auc=m["roc_auc"],
                         val_sens=m["sensitivity"], val_spec=m["specificity"]))
        print(f"ep {ep:03d} | loss {trl:.4f}/{vll:.4f} | val acc {m['accuracy']:.4f} auc {m['roc_auc']:.4f} "
              f"sens {m['sensitivity']:.4f} spec {m['specificity']:.4f} | {time.time() - t0:.0f}s")
        if auc > best:
            best, bad = auc, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": auc,
                        "config": {"img_size": args.size, "arch": "efficientnet_b0"}}, args.out)
            print(f"  saved best -> {args.out} (val AUC {best:.4f})")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
        pd.DataFrame(hist).to_csv(config.REPORTS_DIR / "cls_history.csv", index=False)
    plot_history(pd.DataFrame(hist), [["train_loss", "val_loss"], ["val_acc", "val_auc", "val_sens", "val_spec"]],
                 config.REPORTS_DIR / "cls_training_curves.png", "EfficientNet-B0 training")
    print(f"done. best val AUC = {best:.4f}")


if __name__ == "__main__":
    main()
