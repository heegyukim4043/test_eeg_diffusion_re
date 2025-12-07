# train_subject.py
import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset_subject import EEGImageSubjectDataset
from model import EEGDiffusionModel


def train_subject(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ---------- Dataset & Dataloader ----------
    train_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=args.subject_id,
        split="train",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # EEG 채널 수 자동 추론
    sample_eeg, sample_img, _ = train_ds[0]
    eeg_channels = sample_eeg.shape[0]
    print(f"EEG channels: {eeg_channels}, img size: {sample_img.shape[-2:]}")

    # ---------- Model ----------
    model = EEGDiffusionModel(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        eeg_hidden_dim=args.eeg_hidden_dim,
        time_dim=args.time_dim,
        base_channels=args.base_channels,
        num_timesteps=args.num_timesteps,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0

    # ---------- Training Loop ----------
    model.train()
    for epoch in range(args.epochs):
        for eeg, img, label in train_loader:
            eeg = eeg.to(device)                    # (B, 32, 512)
            img = img.to(device)                    # (B, 3, H, W)
            img = img * 2.0 - 1.0                   # [0,1] -> [-1,1]

            b = img.size(0)
            t = torch.randint(
                0, args.num_timesteps, (b,), device=device
            ).long()

            loss = model.p_losses(img, eeg, t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step % args.log_interval == 0:
                print(f"Epoch {epoch} Step {global_step} Loss {loss.item():.4f}")

            if global_step > 0 and global_step % args.ckpt_interval == 0:
                ckpt_path = os.path.join(args.out_dir, f"subj{args.subject_id:02d}_step{global_step}.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")

            global_step += 1

    # 마지막 체크포인트
    final_ckpt = os.path.join(args.out_dir, f"subj{args.subject_id:02d}_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": global_step,
        },
        final_ckpt,
    )
    print(f"Training finished. Final checkpoint: {final_ckpt}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")
    p.add_argument("--subject_id", type=int, default=1)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_timesteps", type=int, default=200)  # 일단 200으로 가볍게
    p.add_argument("--eeg_hidden_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--split_ratio", type=float, default=0.7)
    p.add_argument("--out_dir", type=str, default="./checkpoints_subj")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_subject(args)
