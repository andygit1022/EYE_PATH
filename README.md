# 프로젝트 소개

이 리포지토리는 YOLOv11 모델을 이용하여 객체 인식을 수행합니다.

## 디렉토리 구조
- `data/` : 데이터 관련 처리 (CSV 개요 파악, train/val split, ...)
- `train/` : 실제 사용된 훈련 코드
- `test/` : 학습된 모델을 이용해 추론(테스트)하는 로직
- `utils/` : 전역 파라미터(PARAMS.py) 등 공통 유틸

## 사용 방법
1. `requirements.txt` 설치  
