# src/test/run_realtime_test.py

"""
'실시간' 또는 '반자동' 테스트 예시:
- 폴더 내 여러 이미지를 순회하며 추론
- 혹은 웹캠(비디오)로부터 프레임을 받아 추론(예시)
"""

import os
import glob
import cv2
from ultralytics import YOLO

def run_test_on_folder(model_path, folder_path, out_dir="realtime_test_results"):
    """
    folder_path 내 모든 jpg/png에 대해 추론 수행 → 결과 시각화 저장
    """
    model = YOLO(model_path)
    os.makedirs(out_dir, exist_ok=True)

    img_list = glob.glob(os.path.join(folder_path, "*.jpg")) + \
               glob.glob(os.path.join(folder_path, "*.png"))

    for img_path in img_list:
        # 추론
        results = model.predict(source=img_path)
        # results[0].plot() 으로 numpy array 결과 얻을 수 있음
        # 시각화한 이미지를 파일로 저장

        # 예시 코드
        res_img = results[0].plot()
        base_name = os.path.basename(img_path)
        save_path = os.path.join(out_dir, base_name)
        cv2.imwrite(save_path, res_img)
        print(f"Saved: {save_path}")

def run_test_webcam(model_path, camera_index=0):
    """
    웹캠(또는 동영상) 실시간 추론 예시 (Linux GUI 필요)
    - SSH로 접속 중이면 화면 표시가 어려우므로 VNC나 X11 포워딩 등 필요
    - 여기서는 개념 예시
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Realtime Inference", annotated_frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
