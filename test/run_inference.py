# src/test/run_inference.py

"""
훈련 완료된 모델(.pt 등)을 불러와서
임의의 이미지(또는 폴더)에 대해 객체 인식을 수행하는 스크립트
"""
# src/test/run_inference.py

"""
학습된 모델(.pt)을 불러와서
단일 이미지나 여러 이미지를 추론 후 결과 리턴/저장
"""

import os
import cv2
from models.YOLO_V8 import infer_yolov8
from ultralytics import YOLO

def run_inference_on_image(model_path, image_path, save_dir="inference_results"):
    """
    model_path: 학습 완료된 가중치 (ex: best.pt)
    image_path: 추론할 이미지 경로
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, save=True, project=save_dir, name="inf_images")
    return results

def run_inference_yolov8(model_path, image_path):
    """
    YOLOv8 가중치(model_path)를 로드해서 image_path를 추론
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model.predict(source=image_path)
    return results

def run_inference_yolov11(model_path, image_path):
    """
    YOLOv11 (가정) 모델 가중치(model_path)를 로드해서 image_path를 추론
    """
    # TODO: YOLOv11 모델 불러오기
    pass

def test_on_folder(model_path, folder_path, out_dir="inference_results"):
    """
    folder_path 내의 모든 이미지에 대해 inference,
    결과를 out_dir에 저장하는 예시
    """
    os.makedirs(out_dir, exist_ok=True)
    # TODO: 실제 구현
    pass
