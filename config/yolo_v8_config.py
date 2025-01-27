# src/config/yolo_config.py

"""
Ultralytics YOLO (v8, v11 등 공용)에서 사용할 기본 설정 모음.
굳이 분리하지 않고, PARAMS.py에 두어도 되지만
별도 config로 두고 싶다면 이렇게 해둠.
"""

import os

def get_yolo_training_params(params_py_dict):
    """
    PARAMS.py 에서 불러온 딕셔너리를 바탕으로
    Ultralytics YOLO의 train config를 구성해 반환
    """
    config = {
        "model": params_py_dict["MODEL"],      # "yolov8n.pt"
        "epochs": params_py_dict["EPOCHS"],    # 50
        "batch": params_py_dict["BATCH_SIZE"], # 8
        "lr0": params_py_dict["LR"],           # 1e-3
        "imgsz": params_py_dict["IMG_SIZE"],   # 640
        # ... 필요시 추가
    }
    return config
