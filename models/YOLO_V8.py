# src/models/YOLO_V8.py

from ultralytics import YOLO

def load_yolov8_model(pretrained_weight):
    """
    Ultralytics YOLOv8 모델 로드
    """
    model = YOLO(pretrained_weight)
    return model

def train_yolov8(model, config_dict):
    """
    Ultralytics YOLOv8 모델 학습 (간단 예시)
    config_dict: PARAMS.py에서 가져온 파라미터
    """
    results = model.train(
        data=config_dict["data_yaml"],  # yolo_dataset/data.yaml 등
        epochs=config_dict["epochs"],
        batch=config_dict["batch"],
        imgsz=config_dict["imgsz"],
        lr0=config_dict["lr0"],
        project=config_dict["project"],
        name=config_dict["exp_name"],
        exist_ok=True
    )
    return results

def infer_yolov8(model, image_path):
    results = model.predict(source=image_path)
    return results
