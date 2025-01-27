# 프로젝트 소개

이 리포지토리는 YOLOv8, YOLOv11(가정) 모델을 이용하여 객체 인식을 수행하는 예시입니다.

## 디렉토리 구조
- `config/` : 모델 훈련에 필요한 하이퍼파라미터 및 설정 (yolo_v8_config.py, yolo_v11_config.py)
- `data/` : 데이터 관련 처리 (CSV 개요 파악, train/val split, ...)
- `models/` : 각 모델별 로드/학습/추론 함수
- `test/` : 학습된 모델을 이용해 추론(테스트)하는 로직
- `train/` : 실제 훈련 파이프라인
- `utils/` : 전역 파라미터(PARAMS.py) 등 공통 유틸
- `main.py` : 전체 실행 진입점

## 사용 방법 (예시)
1. `requirements.txt` 설치  
