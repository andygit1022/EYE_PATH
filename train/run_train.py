# src/train/run_train.py

import os
import matplotlib.pyplot as plt

from models.YOLO_V8 import load_yolov8_model
from utils.PARAMS import WORKSPACE_DIR, MODEL, BATCH_SIZE, EPOCHS, IMG_SIZE, LR
from utils.PARAMS import DATA_CSV_PATH
from config.yolo_config import get_yolo_training_params

def run_training_yolov8():
    """
    YOLOv8 학습을 예시로:
    - 1 epoch씩 루프 돌면서 train_loss, val_loss 기록
    - 매 epoch 끝날 때마다 plot 갱신, 저장
    """

    # 1) 모델 로드
    model = load_yolov8_model(pretrained_weight=MODEL)

    # 2) 학습 설정 (Ultralytics config)
    base_config = {
        "epochs": 1,  # 한 번에 1 epoch씩
        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        "lr0": LR,
        "project": WORKSPACE_DIR,
        "exp_name": "yolov8_exp",
        "data_yaml": "yolo_dataset/data.yaml",  # 예: 사전에 만들어둔 data.yaml
    }

    # 3) 루프 돌며 loss 기록
    train_loss_list, val_loss_list = [], []

    for epoch in EPOCHS:
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Start Training...")
        results = model.train(**base_config)  # 1 epoch 진행

        # results.metrics: Ultralytics 8.x에서 반환되는 학습 결과
        # (버전마다 다를 수 있으니 실제 metrics를 print로 확인해보세요)
        # 여기서는 예시로 train_loss, val_loss를 추정
        # ref: results[0].metrics
        #   - 'train/box_loss', 'train/cls_loss', 'metrics/precision', ...
        #   - 'val/box_loss', 'val/obj_loss', ...
        # 아래는 대략적 예시:

        try:
            r = results  # Usually a list or object
            train_loss = r.metrics.get('train/box_loss', 0) + r.metrics.get('train/cls_loss', 0)
            val_loss   = r.metrics.get('val/box_loss', 0)   + r.metrics.get('val/cls_loss', 0)
        except:
            train_loss = 0
            val_loss   = 0

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # 4) 플롯 그려서 저장
        plt.figure(figsize=(6,4))
        plt.plot(train_loss_list, label="Train Loss")
        plt.plot(val_loss_list, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/loss_curve_epoch_{epoch+1}.png")
        plt.close()

    print("Training completed. Check 'plots/' directory for loss curves.")
