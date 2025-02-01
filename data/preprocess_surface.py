import os
import xml.etree.ElementTree as ET
import random
import shutil
from glob import glob

def parse_xml_to_yolo_seg(xml_path, class2id_dict):
    """
    단일 XML 파일을 읽어, 해당 XML에 포함된 <polygon> 요소들을
    YOLO 세그멘테이션 형식으로 변환하여 반환합니다.
    
    반환 형식:
      {
        "이미지이름.jpg": [
          (class_id, x_center, y_center, w, h, [(px1, py1), (px2, py2), ...]),
          (class_id, x_center, y_center, w, h, [(px1, py1), (px2, py2), ...]),
          ...
        ],
        ...
      }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations_per_image = {}

    # <image ...> 노드를 순회하며, 내부 <polygon> 태그들을 파싱
    for image_tag in root.findall('image'):
        image_name = image_tag.get('name')
        img_width = float(image_tag.get('width'))
        img_height = float(image_tag.get('height'))

        polygons_info = []

        for poly_tag in image_tag.findall('polygon'):
            label = poly_tag.get('label')
            if label not in class2id_dict:
                class2id_dict[label] = len(class2id_dict)
            class_id = class2id_dict[label]

            # points="x1,y1;x2,y2;..."를 파싱
            points_str = poly_tag.get('points')  # 예) "0.00,678.31;289.62,217.79;..."
            # 문자열을 ';'로 split → 각 좌표를 ','로 split
            point_pairs = points_str.strip().split(';')
            polygon_points = []
            for pair in point_pairs:
                x_str, y_str = pair.split(',')
                x_f = float(x_str)
                y_f = float(y_str)
                polygon_points.append((x_f, y_f))

            # 폴리곤의 Bounding Box 계산 (min_x, max_x, min_y, max_y)
            xs = [p[0] for p in polygon_points]
            ys = [p[1] for p in polygon_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # YOLO BBox (x_center, y_center, w, h) [정규화]
            bbox_xcenter = (min_x + max_x) / 2.0 / img_width
            bbox_ycenter = (min_y + max_y) / 2.0 / img_height
            bbox_w = (max_x - min_x) / img_width
            bbox_h = (max_y - min_y) / img_height

            # 폴리곤 좌표들도 0~1로 정규화
            norm_poly_points = []
            for (px, py) in polygon_points:
                nx = px / img_width
                ny = py / img_height
                norm_poly_points.append((nx, ny))

            # (class_id, x_center, y_center, w, h, [ (x1, y1), (x2, y2), ... ])
            polygons_info.append((class_id, bbox_xcenter, bbox_ycenter, bbox_w, bbox_h, norm_poly_points))

        if polygons_info:
            annotations_per_image[image_name] = polygons_info
        else:
            # 폴리곤이 하나도 없는 이미지라면, 빈 리스트로라도 등록할 수 있음
            annotations_per_image[image_name] = []

    return annotations_per_image


def main():
    """
    1) 'Surface_####' 디렉토리들을 무작위로 train/val 분할
    2) train_val_split.txt 파일로 분류 결과 기록
    3) 각 XML을 파싱하여 YOLO 세그멘테이션 라벨(txt) 생성 & 이미지 복사
    4) 진행률(%) 콘솔 표시
    5) dataset.yaml 생성
    """

    # ----------------------------------------------------
    # 1) 기본 설정
    # ----------------------------------------------------
    # (1) root, Surface 폴더 위치, dataset 디렉토리 지정
    ROOT_DIR = "/mnt/walking"        # 실제 프로젝트 경로
    SURFACE_DIR = os.path.join(ROOT_DIR, "surface")  # 예: "/path/to/walking/surface"
    DATASET_DIR = "/mnt/dataset_surface"     # 최종 결과를 저장할 dataset 폴더

    # (2) train/val 비율, 랜덤 시드
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    RANDOM_SEED = 2023

    # (3) 클래스 -> class_id 매핑 딕셔너리
    class2id_dict = {}

    # ----------------------------------------------------
    # 2) dataset 디렉토리 구조 생성
    # ----------------------------------------------------
    os.makedirs(os.path.join(DATASET_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "val"), exist_ok=True)

    # ----------------------------------------------------
    # 3) Surface_#### 디렉토리 목록 -> Train/Val 무작위 분할
    # ----------------------------------------------------
    surface_dirs = sorted(glob(os.path.join(SURFACE_DIR, "Surface_*")))
    random.seed(RANDOM_SEED)
    random.shuffle(surface_dirs)

    num_train = int(len(surface_dirs) * TRAIN_RATIO)
    train_dirs = surface_dirs[:num_train]
    val_dirs = surface_dirs[num_train:]

    # ----------------------------------------------------
    # (추가) train/val 분류 결과 텍스트 저장
    # ----------------------------------------------------
    split_info_path = os.path.join(DATASET_DIR, "train_val_split.txt")
    with open(split_info_path, 'w', encoding='utf-8') as f:
        f.write(f"Train/Val Split Info\n")
        f.write(f"--------------------\n")
        f.write(f"Train Ratio: {TRAIN_RATIO}\n")
        f.write(f"Val Ratio:   {VAL_RATIO}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n\n")

        f.write(f"Train Directories ({len(train_dirs)}):\n")
        for d in train_dirs:
            f.write(f"  {os.path.basename(d)}\n")
        f.write("\n")

        f.write(f"Val Directories ({len(val_dirs)}):\n")
        for d in val_dirs:
            f.write(f"  {os.path.basename(d)}\n")

    # ----------------------------------------------------
    # 4) 폴리곤 세그멘테이션 라벨 생성 & 이미지 복사 (진행률 표시)
    # ----------------------------------------------------
    # (1) 전체 .jpg 파일 개수를 미리 계산
    all_jpg_files = []
    for sdir in surface_dirs:
        all_jpg_files.extend(glob(os.path.join(sdir, "*.jpg")))
    total_jpg_count = len(all_jpg_files)

    processed_count = 0

    # (2) 실제 처리
    for dirs, subset in [(train_dirs, "train"), (val_dirs, "val")]:
        for sdir in dirs:
            # Surface_#### 내부에 있는 xml, jpg 찾기
            xml_files = glob(os.path.join(sdir, "*.xml"))
            jpg_files = glob(os.path.join(sdir, "*.jpg"))

            if not xml_files:
                # XML이 없다면: 그냥 이미지만 복사(라벨=빈 파일 or 미생성)
                for img_file in jpg_files:
                    processed_count += 1
                    # 진행률
                    progress = (processed_count / total_jpg_count) * 100
                    print(f"[{processed_count}/{total_jpg_count}] ({progress:.2f}%)", end='\r')

                    # 이미지 복사
                    dst_img_path = os.path.join(DATASET_DIR, "images", subset, os.path.basename(img_file))
                    shutil.copy2(img_file, dst_img_path)
                continue

            # 보통 1개의 Surface_#### 폴더당 XML 파일 1개라고 가정
            xml_file = xml_files[0]

            # XML 파싱 (세그멘테이션)
            annotations_dict = parse_xml_to_yolo_seg(xml_file, class2id_dict)

            # jpg_files 각각에 대해 라벨(txt) 생성 & 이미지 복사
            for img_file in jpg_files:
                img_name = os.path.basename(img_file)

                # XML에 해당 이미지 정보가 있는지 확인
                if img_name not in annotations_dict:
                    # 폴리곤이 하나도 없는 경우 => 빈 리스트
                    polygons = []
                else:
                    polygons = annotations_dict[img_name]

                # 라벨 txt 경로
                txt_name = os.path.splitext(img_name)[0] + ".txt"
                txt_path = os.path.join(DATASET_DIR, "labels", subset, txt_name)

                # txt 파일 생성
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for (class_id, xc, yc, w, h, poly_pts) in polygons:
                        # YOLO 세그멘테이션 형식
                        #   class_id x_center y_center w h x1 y1 x2 y2 ... xN yN
                        line_elems = [
                            str(class_id),
                            f"{xc:.6f}", f"{yc:.6f}",
                            f"{w:.6f}", f"{h:.6f}",
                        ]
                        # 폴리곤 점들 추가 (x1, y1, x2, y2, ...)
                        for (nx, ny) in poly_pts:
                            line_elems.append(f"{nx:.6f}")
                            line_elems.append(f"{ny:.6f}")

                        line_str = " ".join(line_elems)
                        f.write(line_str + "\n")

                # 이미지 복사
                dst_img_path = os.path.join(DATASET_DIR, "images", subset, img_name)
                shutil.copy2(img_file, dst_img_path)

                # 진행률 갱신
                processed_count += 1
                progress = (processed_count / total_jpg_count) * 100
                print(f"[{processed_count}/{total_jpg_count}] ({progress:.2f}%)", end='\r')

    # 줄바꿈
    print()

    # ----------------------------------------------------
    # 5) dataset.yaml 파일 생성
    # ----------------------------------------------------
    # 세그멘테이션용 YAML이라고 해도 기본 구조는 동일 (train, val, nc, names)
    sorted_items = sorted(class2id_dict.items(), key=lambda x: x[1])
    class_names = [k for k, v in sorted_items]

    yaml_text = f"""train: {os.path.join(DATASET_DIR, 'images', 'train')}
val: {os.path.join(DATASET_DIR, 'images', 'val')}

# number of classes
nc: {len(class_names)}

# class names
names: {class_names}
"""
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_text)

    print("----- 완료 -----")
    print(f"생성된 클래스 ID 매핑: {class2id_dict}")
    print(f"dataset.yaml 생성 위치: {yaml_path}")
    print(f"Train/Val 분류 결과: {split_info_path}")


if __name__ == "__main__":
    main()
