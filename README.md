# EEG Diffusion (per-subject)

EEG 신호와 시각 자극 이미지를 함께 사용해 조건부 확산 모델을 학습/샘플링하는 실험 코드입니다. 데이터가 크기 때문에 저장소에는 `preproc_data/sample.mat`만 포함되어 있으며, 실제 학습용 데이터(`subj_XX.mat`)가 없을 경우 자동으로 샘플 데이터를 사용합니다.

## 데이터 구조
- EEG/이미지 데이터 루트: `preproc_data/`
- 기본 파일 기대치: `subj_XX.mat` (예: `subj_01.mat`)
  - 원본 형태: `X` → `(32, 512, 540)`, `y` → `(540, 1)` (ch × time × trial)
- 샘플 파일: `sample.mat` (18 trial) – 기본 파일이 없을 때 자동 사용
- 시각 자극 이미지: `preproc_data/images/01.png` ~ `09.png` (라벨 1~9)

## 주요 스크립트
- `dataset_subject.py`: EEG/이미지를 묶어 한 명의 subject에 대한 PyTorch `Dataset` 제공
- `train_subject.py`: 단일 subject 기준 학습 스크립트 (기본 70/30 train/test split)
- `sample_subject.py`: 학습된 체크포인트로 DDIM 스타일 샘플 생성
- `test_dataset_subject01.py`: 데이터셋 로드/배치 구성이 정상인지 빠른 스모크 테스트

## 사용 예시
### 1) 데이터셋 동작 확인
```bash
python test_dataset_subject01.py
```
`subj_XX.mat`가 없으면 `sample.mat`를 사용하며, 학습/테스트 분할 및 텐서 크기를 출력합니다.

### 2) 학습
```bash
python train_subject.py --data_root ./preproc_data --subject_id 1 --epochs 100
```
- 이미지 값은 `[0,1]` → `[-1,1]`으로 변환되어 확산 손실을 계산합니다.
- 체크포인트는 `checkpoints_subj/subjXX_*.pt`로 저장됩니다.

### 3) 샘플링
```bash
python sample_subject.py --data_root ./preproc_data --subject_id 1 --trial_index 0 \
  --ckpt_path ./checkpoints_subj/subj01_final.pt --num_timesteps 200
```
- `trial_index`에 해당하는 EEG를 조건으로 이미지를 생성하고, GT 이미지도 함께 저장합니다.

## 개발 메모
- `EEGImageSubjectDataset`는 생성자 인자로 `mat_path`를 받아 임의의 MAT 파일을 지정할 수 있습니다. 인자가 없고 기본 파일이 없을 때는 `sample.mat`로 대체합니다.
- 재현성을 위해 `seed`로 NumPy 셔플을 고정합니다.
- 학습 시 그래디언트 클리핑(`max_norm=1.0`)을 적용해 폭주를 방지합니다.
