# src/data/data_overview.py

"""
데이터 전체 정보 파악:
- CSV 파일에서 읽은 전체 라인 수
- 각 Bbox_xxxx 디렉토리가 몇 개 있는지
- 라벨별 개수 등을 출력
"""

import csv
import os
from collections import defaultdict

def overview_data(csv_path: str):
    """
    CSV 파일 경로를 받아서:
    1) 전체 행(row) 개수
    2) Bbox_xxxx 디렉토리(unique) 개수
    3) 라벨(label)별 개수
    등을 출력/반환
    """
    total_rows = 0
    bbox_dir_set = set()
    label_count = defaultdict(int)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # 예시: 278, Bbox_0345/MP_SEL_067785.jpg, car, 1, 208.0, ...
            total_rows += 1
            image_path = row[1]  # ex) "Bbox_0345/MP_SEL_067785.jpg"
            label_name = row[2]

            # Bbox_xxxx 추출
            # ex) image_path.split('/')[0] => "Bbox_0345"
            bbox_dir_set.add(image_path.split('/')[0])

            # 라벨 카운팅
            label_count[label_name] += 1

    print(f"Total rows in CSV: {total_rows}")
    print(f"Number of unique Bbox dirs: {len(bbox_dir_set)}")
    print("Label distribution:")
    for lbl, cnt in label_count.items():
        print(f"  {lbl}: {cnt}")

    # 필요하다면 리턴도 가능
    return {
        "total_rows": total_rows,
        "num_bbox_dirs": len(bbox_dir_set),
        "label_count": dict(label_count)
    }
