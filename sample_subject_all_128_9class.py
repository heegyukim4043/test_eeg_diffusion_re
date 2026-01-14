# sample_subject_all_128_9class.py
import os
import argparse

import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model_128 import EEGDiffusionModel128


# ---------------------------------------------------------
# 1. Dataset: 한 subject, 9-class 전체, "test index"만 사용
#    - 데이터 파일: ./preproc_data/subj_16.mat 형태
#    - mat 구조: X (ch, time, trial), y (trial,)
# ---------------------------------------------------------
class EEGImageDataset9Class128(Dataset):
    def __init__(
        self,
        mat_path: str,
        img_root: str,
        indices,
        img_size: int = 128,
    ):
        super().__init__()
        self.mat_path = mat_path
        self.img_root = img_root
        self.indices = np.array(indices, dtype=np.int64)
        self.img_size = img_size

        mat = loadmat(mat_path)
        X = mat["X"]          # (ch, time, trial)
        y = mat["y"].squeeze()  # (trial,)

        # (trial, ch, time)
        self.eeg = torch.from_numpy(X).float().permute(2, 0, 1)
        self.labels = y.astype(np.int64)  # 1~9

        # 이미지: class별 GT PNG (예: images/01.png ~ images/09.png)
        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),  # [0,1]
                T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),  # [-1,1]로 정규화
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trial_idx = int(self.indices[idx])

        eeg = self.eeg[trial_idx]  # (C, T)
        label = int(self.labels[trial_idx])  # 1~9

        # class별 GT 이미지 한 장 (훈련 때와 동일한 방식)
        img_path = os.path.join(self.img_root, f"{label:02d}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # (3,H,W), [-1,1]

        return eeg, img, label, trial_idx


# ---------------------------------------------------------
# 2. 시드 고정 (train과 동일 seed 재사용용)
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# 3. train_subject_128_9class.py와 동일한 방식으로
#    train/val/test split 복원 (8:1:1)
# ---------------------------------------------------------
def get_test_indices_9cls(
    mat_path: str,
    subject_id: int,
    seed: int,
):
    """
    - 전체 trial을 8:1:1로 나누고, test index만 반환
    - train_subject_128_9class.py에서 사용한 로직과 동일하게 맞추기 위해
      RandomState(seed + subject_id*10) 사용
    """
    mat = loadmat(mat_path)
    y = mat["y"].squeeze().astype(np.int64)
    n_total = len(y)

    all_indices = np.arange(n_total, dtype=np.int64)

    rng = np.random.RandomState(seed + subject_id * 10)
    rng.shuffle(all_indices)

    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:n_train + n_val]
    test_idx = all_indices[n_train + n_val:]

    return test_idx


# ---------------------------------------------------------
# 4. subject 1개에 대한 "가장 최근" 9-class checkpoint 디렉토리 찾기
#    예: 20260112_221228_subj16_9cls_128
# ---------------------------------------------------------
def find_latest_9cls_ckpt_dir(ckpt_root, subject_id, img_size):
    subj_str = f"{subject_id:02d}"
    target_suffix = f"_subj{subj_str}_9cls_{img_size}"

    if not os.path.isdir(ckpt_root):
        return None

    cand_dirs = []
    for name in os.listdir(ckpt_root):
        full = os.path.join(ckpt_root, name)
        if not os.path.isdir(full):
            continue
        if name.endswith(target_suffix):
            cand_dirs.append(full)

    if not cand_dirs:
        return None

    cand_dirs.sort()
    return cand_dirs[-1]  # 가장 최근(timestamp가 큰 것)


# ---------------------------------------------------------
# 5. main: test 전체 trial에 대해 (eeg → 이미지) 생성 + GT 저장
# ---------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_str = f"{args.subject_id:02d}"

    print("=" * 80)
    print(f"[Sample-9cls] Subject {subj_str}, device: {device}")
    print("=" * 80)

    # ------------------ 데이터 경로 ------------------
    mat_path = os.path.join(args.data_root, f"subj_{subj_str}.mat")
    img_root = os.path.join(args.data_root, "images")

    # ------------------ test index 복원 ------------------
    set_seed(args.seed)
    test_idx = get_test_indices_9cls(
        mat_path=mat_path,
        subject_id=args.subject_id,
        seed=args.seed,
    )
    if len(test_idx) == 0:
        print(f"[Subj {subj_str}] No test trials found. 종료.")
        return

    print(f"[Subj {subj_str}] number of test trials: {len(test_idx)}")

    # ------------------ Dataset / DataLoader ------------------
    test_ds = EEGImageDataset9Class128(
        mat_path=mat_path,
        img_root=img_root,
        indices=test_idx,
        img_size=args.img_size,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------ Checkpoint 디렉토리 찾기 ------------------
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = find_latest_9cls_ckpt_dir(
            args.ckpt_root,
            args.subject_id,
            args.img_size,
        )

    if ckpt_dir is None or (not os.path.isdir(ckpt_dir)):
        print(
            f"[Subj {subj_str}] Checkpoint dir not found. "
            f"ckpt_root={args.ckpt_root}"
        )
        return

    best_path = os.path.join(ckpt_dir, f"subj{subj_str}_best.pt")
    final_path = os.path.join(ckpt_dir, f"subj{subj_str}_final.pt")

    if os.path.isfile(best_path):
        ckpt_path = best_path
    elif os.path.isfile(final_path):
        ckpt_path = final_path
    else:
        print(f"[Subj {subj_str}] No best/final checkpoint in {ckpt_dir}")
        return

    print(f"[Subj {subj_str}] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # config 있으면 사용, 없으면 CLI args로
    cfg = ckpt.get("config", {})
    img_size = cfg.get("img_size", args.img_size)
    base_channels = cfg.get("base_channels", args.base_channels)
    num_timesteps = cfg.get("num_timesteps", args.num_timesteps)
    n_res_blocks = cfg.get("n_res_blocks", args.n_res_blocks)

    # ------------------ 모델 구성 ------------------
    model = EEGDiffusionModel128(
        img_size=img_size,
        img_channels=3,
        eeg_channels=32,
        num_classes=9,            # 9-class 전체
        num_timesteps=num_timesteps,
        base_channels=base_channels,
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=n_res_blocks,
    ).to(device)

    state_dict = ckpt.get("ema", ckpt.get("model", None))
    if state_dict is None:
        raise KeyError("Checkpoint에 'model' 또는 'ema' state_dict가 없습니다.")
    model.load_state_dict(state_dict)
    model.eval()

    # ------------------ 저장 폴더 ------------------
    os.makedirs(args.samples_root, exist_ok=True)
    ckpt_basename = os.path.basename(ckpt_dir.rstrip("/\\"))
    samples_dir = os.path.join(args.samples_root, ckpt_basename)
    os.makedirs(samples_dir, exist_ok=True)

    print(f"Samples will be saved under: {samples_dir}")

    # ------------------ Helper: [-1,1] → [0,1] ------------------
    def denorm_img(x):
        return (x.clamp(-1, 1) + 1.0) * 0.5

    to_pil = T.ToPILImage()

    # ------------------ sampling loop ------------------
    with torch.no_grad():
        global_sample_idx = 0

        for batch_idx, (eeg, img_gt, labels, trial_idx) in enumerate(test_loader):
            eeg = eeg.to(device)          # (B,C,T)
            img_gt = img_gt.to(device)    # (B,3,H,W)
            labels = labels.to(device)    # (B,), 1~9

            b = eeg.size(0)

            x_gen = model.sample(
                eeg=eeg,
                labels=labels,
                num_steps=args.sample_steps,
            )  # (B,3,H,W), [-1,1]

            x_gen_denorm = denorm_img(x_gen)
            img_gt_denorm = denorm_img(img_gt)

            for i in range(b):
                g_label = int(labels[i].item())       # 1~9
                t_idx = int(trial_idx[i].item())

                gen_pil = to_pil(x_gen_denorm[i].cpu())
                gt_pil = to_pil(img_gt_denorm[i].cpu())

                gen_name = (
                    f"subj{subj_str}_trial{t_idx:03d}_"
                    f"label{g_label}_GEN.png"
                )
                gt_name = (
                    f"subj{subj_str}_trial{t_idx:03d}_"
                    f"label{g_label}_GT.png"
                )

                gen_path = os.path.join(samples_dir, gen_name)
                gt_path = os.path.join(samples_dir, gt_name)

                gen_pil.save(gen_path)
                gt_pil.save(gt_path)

                global_sample_idx += 1

                if global_sample_idx % 10 == 0:
                    print(
                        f"[Subj {subj_str}] "
                        f"Saved {global_sample_idx} samples so far..."
                    )

    print(
        f"[Subj {subj_str}] Done. "
        f"Total saved files: {global_sample_idx * 2} "
        f"(generated + GT)."
    )


# ---------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate images for ALL test trials, "
            "for a given subject (9-class, 128x128)."
        )
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--subject_id", type=int, required=True,
                        help="예: 16 (→ subj_16.mat)")
    parser.add_argument("--img_size", type=int, default=128)

    parser.add_argument("--ckpt_root", type=str,
                        default="./checkpoints_subj128_9cls",
                        help="train_subject_128_9class.py에서 사용한 ckpt_root")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="특정 ckpt 디렉토리를 직접 지정할 때 사용")

    parser.add_argument("--samples_root", type=str,
                        default="./samples_subj128_9cls")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--num_timesteps", type=int, default=275)
    parser.add_argument("--n_res_blocks", type=int, default=7)
    parser.add_argument("--sample_steps", type=int, default=275,
                        help="sampling에 사용할 step 수 (<= num_timesteps 권장)")
    parser.add_argument("--seed", type=int, default=42,
                        help="train_subject_128_9class에서 사용한 seed와 동일하게")

    args = parser.parse_args()
    main(args)
