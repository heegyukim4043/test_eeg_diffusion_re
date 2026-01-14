import os
import glob
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from model_128 import EEGDiffusionModel128


# ---------------------------------------------------------
# 1. Dataset (9-class, 128x128)
#    - preproc_data/subjXX.npz 에서 eeg, 이미지, label 로드
# ---------------------------------------------------------
class EEGImageDataset9Class(Dataset):
    """
    NPZ 구조 예시 (필요시 키 이름만 맞춰 수정):
      - 'eeg'   : (N, C, T)
      - 'images' 또는 'img' : (N, 3, H, W) 또는 (N, H, W, 3)
      - 'labels' 또는 'y'   : (N,)
    """

    def __init__(self, data_root, subject_id, indices, img_size=128):
        super().__init__()
        npz_path = os.path.join(data_root, f"subj{subject_id:02d}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        self.data_root = data_root
        self.subject_id = subject_id
        self.img_size = img_size

        npz = np.load(npz_path)

        # ---- 키 이름 유연하게 처리 ----
        if "eeg" in npz:
            eeg = npz["eeg"]
        elif "EEG" in npz:
            eeg = npz["EEG"]
        else:
            raise KeyError(f"'eeg' key not found in {npz_path}")

        if "images" in npz:
            img = npz["images"]
        elif "img" in npz:
            img = npz["img"]
        else:
            raise KeyError(f"'images' or 'img' key not found in {npz_path}")

        if "labels" in npz:
            labels = npz["labels"]
        elif "y" in npz:
            labels = npz["y"]
        else:
            raise KeyError(f"'labels' or 'y' key not found in {npz_path}")

        # dtype/shape 정리
        self.eeg = eeg.astype(np.float32)          # (N, C, T)
        self.img = img.astype(np.float32)          # (N, 3, H, W) or (N, H, W, 3)
        self.labels = labels.astype(np.int64)      # (N,)

        # 인덱스 선택
        self.indices = np.array(indices, dtype=np.int64)

        # 이미지 채널 차원 정리 (채널 먼저)
        # (N, H, W, 3) 이면 (N, 3, H, W)로 바꿈
        if self.img.ndim == 4 and self.img.shape[-1] in (1, 3) and self.img.shape[1] != 3:
            # (N, H, W, C) -> (N, C, H, W)
            self.img = np.transpose(self.img, (0, 3, 1, 2))

        if self.img.shape[1] not in (1, 3):
            raise ValueError(
                f"Unexpected image shape: {self.img.shape}. Expected (N,3,H,W) or (N,H,W,3)."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        j = self.indices[idx]

        eeg = torch.from_numpy(self.eeg[j])    # (C, T)
        img = torch.from_numpy(self.img[j])    # (3, H, W) in float
        label = int(self.labels[j])

        # 여기서는 이미지가 이미 [-1,1]로 저장되었다고 가정
        # (만약 [0,1] 이나 [0,255] 라면 시각화 시에만 normalize 해도 무방)
        return eeg, img, label


# ---------------------------------------------------------
# 2. 체크포인트 디렉토리 탐색 (subjXX_9cls_128 중 최신 폴더)
# ---------------------------------------------------------
def find_latest_ckpt_dir(subject_id, ckpt_root="checkpoints_subj128_9cls"):
    pattern = os.path.join(ckpt_root, f"*subj{subject_id:02d}_9cls_128")
    cand = glob.glob(pattern)
    if not cand:
        raise FileNotFoundError(f"No checkpoint dir matches pattern: {pattern}")
    cand.sort()
    return cand[-1]   # 가장 최근(이름/시간 기준 마지막)


# ---------------------------------------------------------
# 3. 메인 로직: test split 전체에 대해 이미지 생성
# ---------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sid = args.subject_id

    print("=" * 80)
    print(f"[Sample-9cls] Subject {sid:02d}, device: {device}")
    print("=" * 80)

    # 1) 최신 체크포인트 디렉토리/파일 선택
    ckpt_dir = find_latest_ckpt_dir(sid, ckpt_root=args.ckpt_root)
    ckpt_path = os.path.join(ckpt_dir, f"subj{sid:02d}_best.pt")
    if not os.path.exists(ckpt_path):
        # 만약 best.pt 이름이 다르면 (예: subj16_best.pt) 여기서 수정
        alt_path = os.path.join(ckpt_dir, "subj16_best.pt")
        if os.path.exists(alt_path):
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"No checkpoint file found in {ckpt_dir}")

    print(f"[Subj {sid:02d}] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # 2) 하이퍼파라미터: ckpt에 hparams가 있으면 우선 사용
    hparams = ckpt.get("hparams", None)

    if hparams is not None:
        print("[Info] Using hparams from checkpoint.")
        img_size = hparams.get("img_size", 128)
        num_timesteps = hparams.get("num_timesteps", 275)
        base_channels = hparams.get("base_channels", 128)
        time_dim = hparams.get("time_dim", 256)
        cond_dim = hparams.get("cond_dim", 256)
        eeg_hidden_dim = hparams.get("eeg_hidden_dim", 256)
        cond_scale = hparams.get("cond_scale", 1.5)
        beta_start = hparams.get("beta_start", 1e-4)
        beta_end = hparams.get("beta_end", 2e-2)
    else:
        print("[Warn] No 'hparams' in checkpoint. Falling back to CLI args.")
        img_size = args.img_size
        num_timesteps = args.num_timesteps
        base_channels = args.base_channels
        time_dim = 256
        cond_dim = 256
        eeg_hidden_dim = 256
        cond_scale = args.cond_scale
        beta_start = 1e-4
        beta_end = 2e-2

    # 3) 모델 생성 및 weight 로딩
    model = EEGDiffusionModel128(
        img_size=img_size,
        img_channels=3,
        eeg_channels=32,
        num_classes=9,
        num_timesteps=num_timesteps,
        base_channels=base_channels,
        time_dim=time_dim,
        cond_dim=cond_dim,
        eeg_hidden_dim=eeg_hidden_dim,
        cond_scale=cond_scale,
        beta_start=beta_start,
        beta_end=beta_end,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # 4) test split 인덱스 계산 (train/val/test = 8:1:1, seed 고정)
    npz_path = os.path.join(args.data_root, f"subj{sid:02d}.npz")
    npz = np.load(npz_path)
    if "labels" in npz:
        N = npz["labels"].shape[0]
    elif "y" in npz:
        N = npz["y"].shape[0]
    else:
        raise KeyError(f"'labels' or 'y' not found in {npz_path}")

    all_idx = np.arange(N)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_idx)

    n_train = int(N * args.split_ratio)          # 기본 0.8
    n_val = int(N * ((1.0 - args.split_ratio) / 2.0))  # 0.1
    n_test = N - n_train - n_val

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]

    print(
        f"[Subj {sid:02d}] total={N}, "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    # 5) Dataset/DataLoader (test split만)
    test_dataset = EEGImageDataset9Class(
        data_root=args.data_root,
        subject_id=sid,
        indices=test_idx,
        img_size=img_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 6) 출력 디렉토리
    out_root = os.path.join(args.sample_root, os.path.basename(ckpt_dir))
    os.makedirs(out_root, exist_ok=True)
    print(f"[Subj {sid:02d}] Samples will be saved to: {out_root}")

    # 7) test 전체에 대해 EEG-conditioned diffusion 샘플 생성
    model.eval()
    torch.set_grad_enabled(False)

    global_idx = 0
    for batch_idx, (eeg, img_gt, labels) in enumerate(test_loader):
        eeg = eeg.to(device)
        labels = labels.to(device)

        # diffusion sampling
        x_gen = model.sample(
            eeg=eeg,
            labels=labels,
            num_steps=args.sample_steps,
        )  # (B,3,H,W), [-1,1]

        # 시각화를 위해 [0,1] 범위로 변환
        x_gen_vis = (x_gen.clamp(-1.0, 1.0) + 1.0) / 2.0
        img_gt_vis = (img_gt.clamp(-1.0, 1.0) + 1.0) / 2.0

        B = eeg.size(0)
        for b in range(B):
            trial_id = global_idx
            label = int(labels[b].item())

            gen_path = os.path.join(
                out_root,
                f"subj{sid:02d}_trial{trial_id:03d}_label{label}_GEN.png",
            )
            gt_path = os.path.join(
                out_root,
                f"subj{sid:02d}_trial{trial_id:03d}_label{label}_GT.png",
            )

            save_image(x_gen_vis[b], gen_path)
            save_image(img_gt_vis[b], gt_path)

            global_idx += 1

        print(
            f"[Subj {sid:02d}] Batch {batch_idx+1}/{len(test_loader)} "
            f"→ saved {B} pairs (GEN/GT)"
        )

    print(f"[Subj {sid:02d}] Done. Total saved trials: {global_idx}")


# ---------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG-conditioned diffusion sampling (9-class, 128x128)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./preproc_data",
        help="Root directory of subjXX.npz",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="./checkpoints_subj128_9cls",
        help="Root directory of 9-class checkpoints",
    )
    parser.add_argument(
        "--sample_root",
        type=str,
        default="./samples_subj128_9cls",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--subject_id",
        type=int,
        required=True,
        help="Subject ID (e.g., 16)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=128,
        help="Image size (should match training)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for sampling",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=275,
        help="Total diffusion steps (only used if hparams not in checkpoint)",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=128,
        help="Base channels of UNet (only used if hparams not in checkpoint)",
    )
    parser.add_argument(
        "--cond_scale",
        type=float,
        default=1.5,
        help="Condition embedding scale (only used if hparams not in checkpoint)",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=0,
        help=(
            "Number of sampling steps. 0 or None → use full num_timesteps from model."
        ),
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Train ratio for train/val/test split (train=ratio, val=test=(1-ratio)/2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting train/val/test",
    )

    args = parser.parse_args()
    main(args)
