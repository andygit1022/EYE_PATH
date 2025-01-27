########### PARAMS.py ##############
import random

# -----------------------------
# 공통 파라미터
# -----------------------------
SEED = 42
random.seed(SEED)

# 이미지 해상도 고정
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# 원본 이미지 루트 디렉토리
ROOT_IMG_DIR = "/mnt/walking/bounding"

# train:test = 0.85 : 0.15
TRAIN_RATIO = 0.85

# 클래스 목록
CLASSES = [
    "bicycle", "bus", "car", "carrier", "cat", "dog", "motorcycle", "movable_signage",
    "person", "scooter", "stroller", "truck", "wheelchair",
    "barricade", "bench", "bollard", "chair", "fire_hydrant", "kiosk",
    "parking_meter", "pole", "potted_plant", "power_controller", "stop",
    "table", "traffic_light", "traffic_light_controller", "traffic_sign", "tree_trunk"
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}

# -----------------------------
# 학습 관련 파라미터
# -----------------------------
MODEL = "yolov8n.pt"    # 사용할 사전 학습 가중치
BATCH_SIZE = 8
EPOCHS = 50
IMG_SIZE = 640  # 전처리/학습 시 입력 이미지 크기(축소)
LR = 1e-3

# -----------------------------
# 경로
# -----------------------------
DATA_CSV_PATH = "src/data/data_info.csv"  # CSV 경로
WORKSPACE_DIR = "yolo_workspace"          # 학습 결과물(가중치, 로그 등) 저장
