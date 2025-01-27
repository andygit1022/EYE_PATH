# src/data/train_split.py

import csv
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils.PARAMS import SEED, TRAIN_RATIO, IMG_WIDTH, IMG_HEIGHT, ROOT_IMG_DIR
from utils.PARAMS import DATA_CSV_PATH, LABEL_TO_IDX

random.seed(SEED)

class YoloDataset(Dataset):
    """
    예시: CSV 정보를 토대로 이미지를 로드하고
    bounding box 라벨( YOLO 포맷 )을 반환하는 Dataset
    - 실제론 이미지 리사이즈, transform 등이 필요
    - 여기서는 개념 예시
    """
    def __init__(self, records, transform=None):
        """
        records: [(image_full_path, label_info_list), ...] 형태
        """
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        img_path, boxes = self.records[index]

        # 이미지를 PIL로 열기
        img = Image.open(img_path).convert("RGB")

        # TODO: 여기서 img 리사이즈/transform 적용 가능
        # bounding box 정규화 등

        # 예시로, x_center, y_center, w, h, class_id 등
        # 한 이미지에 여러 박스가 있을 수 있으나, 여기서는 간단히 반환
        # (실제로는 dict 형태나, batch collate 함수를 별도 작성 필요)
        sample = {
            "image": img,
            "labels": boxes  # [{class_id:..., x_center:..., y_center:..., w:..., h:...}, ...]
        }
        return sample

def split_data_by_bbox(csv_path=DATA_CSV_PATH, train_ratio=TRAIN_RATIO):
    """
    Bbox_xxxx 단위로 train/val 나누고,
    최종 list 반환
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        bbox_set = set()
        for row in reader:
            image_path = row[1]
            bbox_set.add(image_path.split('/')[0])
    bbox_list = list(bbox_set)
    random.shuffle(bbox_list)

    train_count = int(len(bbox_list) * train_ratio)
    train_dirs = bbox_list[:train_count]
    val_dirs = bbox_list[train_count:]

    return train_dirs, val_dirs

def build_dataset(csv_path=DATA_CSV_PATH, bbox_dirs=None):
    """
    주어진 bbox_dirs 목록에 해당하는 이미지+라벨만 모아서
    records 리스트로 만든 다음, YoloDataset으로 감싸서 반환
    """
    # 1) bbox_dir를 set으로
    bbox_dir_set = set(bbox_dirs)

    # 2) CSV 스캔 -> 해당 dir에 속하는 row만 수집
    image_dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # row: [ image_id, image_path, label, occluded, xtl, ytl, xbr, ybr, z_order ]
            image_path = row[1]  # e.g. "Bbox_0345/MP_SEL_067785.jpg"
            label = row[2]
            xtl = float(row[4])
            ytl = float(row[5])
            xbr = float(row[6])
            ybr = float(row[7])

            dir_name = image_path.split('/')[0]
            if dir_name not in bbox_dir_set:
                continue

            full_img_path = os.path.join(ROOT_IMG_DIR, image_path)
            if full_img_path not in image_dict:
                image_dict[full_img_path] = []

            if label not in LABEL_TO_IDX:
                # 범위 밖 라벨이면 스킵(예시)
                continue
            class_id = LABEL_TO_IDX[label]

            # YOLO 포맷 (x_center, y_center, w, h) 정규화
            # 해상도 고정: 1920x1080
            w = xbr - xtl
            h = ybr - ytl
            xc = xtl + w/2
            yc = ytl + h/2

            xcn = xc / IMG_WIDTH
            ycn = yc / IMG_HEIGHT
            wn = w / IMG_WIDTH
            hn = h / IMG_HEIGHT

            image_dict[full_img_path].append({
                "class_id": class_id,
                "x_center": xcn,
                "y_center": ycn,
                "w": wn,
                "h": hn
            })

    # records = [(img_path, [ {class_id, x_center, ...}, ... ]), ...]
    records = list(image_dict.items())
    dataset = YoloDataset(records=records)
    return dataset

def create_data_loaders():
    """
    예시: train/val split -> Dataset -> DataLoader 반환
    """
    train_dirs, val_dirs = split_data_by_bbox()
    train_dataset = build_dataset(bbox_dirs=train_dirs)
    val_dataset   = build_dataset(bbox_dirs=val_dirs)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

    return train_loader, val_loader



# """
# Train/Test split을 담당:
# - Bbox_xxxx 단위로 85:15(or 다른 비율) 분할
# - 분할된 데이터를 실제로 복사/이동/링크 or
#   라벨 txt 생성 등등
# """

# import os
# import csv
# import random
# from collections import defaultdict

# def split_data_by_bbox(csv_path: str, train_ratio: float = 0.85):
#     """
#     1) CSV 읽어서, 각 row에 대해 Bbox_xxxx를 확인
#     2) Bbox_xxxx 단위로 목록을 만들고, shuffle 후 train/val 나눔
#     3) 최종적으로 train에 포함될 Bbox_xxxx, test(val)에 포함될 Bbox_xxxx를 반환
#     """
#     random.seed(42)
#     bbox_dirs = set()

#     with open(csv_path, "r", encoding="utf-8") as f:
#         reader = csv.reader(f)
#         for row in reader:
#             # row[1] => ex) "Bbox_0345/MP_SEL_067785.jpg"
#             bbox_dir = row[1].split('/')[0]
#             bbox_dirs.add(bbox_dir)

#     bbox_dirs = list(bbox_dirs)
#     random.shuffle(bbox_dirs)

#     train_count = int(len(bbox_dirs) * train_ratio)
#     train_dirs = bbox_dirs[:train_count]
#     val_dirs = bbox_dirs[train_count:]

#     print(f"[split_data_by_bbox] Train dirs: {len(train_dirs)}, Val dirs: {len(val_dirs)}")
#     return train_dirs, val_dirs

# def create_yolo_dataset(csv_path: str, train_dirs, val_dirs, output_dir="yolo_dataset"):
#     """
#     실제로 YOLO 포맷에 맞게 train/val set 폴더 구성,
#     라벨 txt파일 생성 등의 로직 작성 (예시).
#     """
#     # 예시 코드(간단 형태). 실제론 bounding box를 (x_center,y_center,w,h)로 변환, etc.
#     # 여기서는 구조만 보여주는 샘플입니다.
#     pass