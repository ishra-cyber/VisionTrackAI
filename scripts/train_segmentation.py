"""Train the U-Net optic disc / optic cup segmentation model (Component 1).

    python scripts/train_segmentation.py --epochs 40 --batch 8
"""
import argparse
import time

import _path  # noqa: F401
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.datasets import SegDataset
from src.models import SegLoss, UNet
from src.utils import plot_history, set_seed


def dice_batch(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) >= thr).float()
    inter = (p * y).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
    d = torch.where(denom > 0, 2 * inter / denom.clamp(min=1e-7), torch.ones_like(denom))
    return d.sum(dim=0)  # per-channel sum over batch


def run_epoch(model, loader, loss_fn, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot_loss, dice_sum, n = 0.0, torch.zeros(2), 0
    use_amp = device == "cuda"
    for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits.float(), y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        tot_loss += loss.item() * x.size(0)
        dice_sum += dice_batch(logits.detach().float(), y).cpu()
        n += x.size(0)
    d = dice_sum / max(n, 1)
    return tot_loss / max(n, 1), d[0].item(), d[1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.SEG_EPOCHS)
    ap.add_argument("--batch", type=int, default=config.SEG_BATCH)
    ap.add_argument("--lr", type=float, default=config.SEG_LR)
    ap.add_argument("--size", type=int, default=config.SEG_IMG_SIZE)
    ap.add_argument("--base", type=int, default=config.SEG_BASE_CH)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--splits", default=str(config.SPLITS_DIR))
    ap.add_argument("--out", default=str(config.SEG_CKPT))
    args = ap.parse_args()

    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    tr = SegDataset(f"{args.splits}/seg_train.csv", args.size, train=True)
    va = SegDataset(f"{args.splits}/seg_val.csv", args.size, train=False)
    print(f"train={len(tr)} val={len(va)}")
    kw = dict(num_workers=args.workers, pin_memory=device == "cuda")
    tl = DataLoader(tr, args.batch, shuffle=True, drop_last=len(tr) > args.batch, **kw)
    vl = DataLoader(va, args.batch, shuffle=False, **kw)

    model = UNet(base=args.base).to(device)
    loss_fn = SegLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 50)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best, bad, hist = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trl, trd, trc = run_epoch(model, tl, loss_fn, device, opt, scaler)
        vll, vd, vc = run_epoch(model, vl, loss_fn, device)
        sched.step()
        score = (vd + vc) / 2
        hist.append(dict(epoch=ep, train_loss=trl, val_loss=vll, train_dice_disc=trd, train_dice_cup=trc,
                         val_dice_disc=vd, val_dice_cup=vc, lr=opt.param_groups[0]["lr"]))
        print(f"ep {ep:03d} | loss {trl:.4f}/{vll:.4f} | val dice disc {vd:.4f} cup {vc:.4f} "
              f"| {time.time() - t0:.0f}s")
        if score > best:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_dice_disc": vd, "val_dice_cup": vc,
                        "config": {"img_size": args.size, "base_ch": args.base}}, args.out)
            print(f"  saved best -> {args.out} (mean dice {best:.4f})")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
        df = pd.DataFrame(hist)
        df.to_csv(config.REPORTS_DIR / "seg_history.csv", index=False)
    plot_history(pd.DataFrame(hist), [["train_loss", "val_loss"], ["val_dice_disc", "val_dice_cup"]],
                 config.REPORTS_DIR / "seg_training_curves.png", "U-Net training")
    print(f"done. best val mean dice = {best:.4f}")


if __name__ == "__main__":
    main()
