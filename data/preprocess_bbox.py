import os
import xml.etree.ElementTree as ET
import random
import shutil
from glob import glob

def parse_xml_to_yolo(xml_path, class2id_dict):
    """
    단일 XML 파일(예: 0704_02.xml)을 읽어서,
    같은 폴더에 들어 있는 이미지들에 대응하는 모든 라벨 정보를
    {이미지이름: [ (class_id, x_center, y_center, w, h), ... ]} 형태로 변환하여 반환.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # image 노드를 순회하면서 <image name="..."> 내부 box들을 파싱
    annotations_per_image = {}
    for image_tag in root.findall('image'):
        image_name = image_tag.get('name')
        img_width = float(image_tag.get('width'))
        img_height = float(image_tag.get('height'))

        boxes = []
        for box_tag in image_tag.findall('box'):
            label = box_tag.get('label')
            if label not in class2id_dict:
                # 아직 class2id_dict에 없으면 추가
                class2id_dict[label] = len(class2id_dict)
            class_id = class2id_dict[label]
            
            xtl = float(box_tag.get('xtl'))
            ytl = float(box_tag.get('ytl'))
            xbr = float(box_tag.get('xbr'))
            ybr = float(box_tag.get('ybr'))

            # YOLO 포맷 변환 (정규화: 0~1 스케일)
            x_center = (xtl + xbr) / 2.0 / img_width
            y_center = (ytl + ybr) / 2.0 / img_height
            w = (xbr - xtl) / img_width
            h = (ybr - ytl) / img_height

            boxes.append((class_id, x_center, y_center, w, h))

        annotations_per_image[image_name] = boxes

    return annotations_per_image


def main():
    """
    1) bounding 폴더(Bbox_####)를 train/val로 랜덤 분할
    2) train_val_split.txt에 분류 결과 기록
    3) YOLO 포맷 라벨(txt) 생성 & 이미지 복사
    4) 진행 상황(%) 콘솔 출력
    5) dataset.yaml 생성
    """

    # ----------------------------------------------------
    # 1) 기본 설정
    # ----------------------------------------------------
    # (1) root(프로젝트) 경로와, bounding 폴더 위치, 그리고 만들고자 하는 dataset 디렉토리 지정
    ROOT_DIR = "/mnt/walking"  # 실제 walking 폴더 상위 경로 예시
    BOUNDING_DIR = os.path.join(ROOT_DIR, "bounding")
    DATASET_DIR = "/mnt/dataset"  # 최종 생성될 dataset 폴더
    
    # (2) train/val 비율, 랜덤 시드
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    RANDOM_SEED = 2023
    
    # YOLO용 클래스 정보 딕셔너리 (label -> class_id)
    class2id_dict = {}

    # ----------------------------------------------------
    # 2) dataset 디렉토리 구조 생성
    # ----------------------------------------------------
    # 이미 존재하면 덮어쓰므로 주의
    os.makedirs(os.path.join(DATASET_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "val"), exist_ok=True)
    
    # ----------------------------------------------------
    # 3) Bbox_#### 디렉토리 목록 -> 무작위 분할
    # ----------------------------------------------------
    bbox_dirs = sorted(glob(os.path.join(BOUNDING_DIR, "Bbox_*")))
    random.seed(RANDOM_SEED)
    random.shuffle(bbox_dirs)
    
    num_train = int(len(bbox_dirs) * TRAIN_RATIO)
    train_dirs = bbox_dirs[:num_train]
    val_dirs = bbox_dirs[num_train:]
    
    # ----------------------------------------------------
    # (추가) train/val 분류 결과를 텍스트로 저장
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
    # 4) YOLO 라벨 생성 & 이미지 복사 (진행률 표시)
    # ----------------------------------------------------
    # 4-1) 전체 .jpg 개수를 미리 계산
    all_jpg_files = []
    for bbox_dir in bbox_dirs:
        all_jpg_files.extend(glob(os.path.join(bbox_dir, "*.jpg")))
    total_jpg_count = len(all_jpg_files)
    
    processed_count = 0  # 지금까지 처리한 jpg 수
    
    # 4-2) train_dirs / val_dirs 순회
    for dirs, subset in [(train_dirs, "train"), (val_dirs, "val")]:
        for bbox_dir in dirs:
            xml_files = glob(os.path.join(bbox_dir, "*.xml"))
            jpg_files = glob(os.path.join(bbox_dir, "*.jpg"))

            if not xml_files:
                # XML이 없다면 스킵
                for img_file in jpg_files:
                    processed_count += 1
                    # 진행률 표시
                    progress = (processed_count / total_jpg_count) * 100
                    print(f"[{processed_count}/{total_jpg_count}] ({progress:.2f}%)", end='\r')
                continue
            
            # 보통 Bbox_#### 폴더당 XML 파일이 1개라고 가정
            xml_file = xml_files[0]
            # XML 파싱
            annotations_dict = parse_xml_to_yolo(xml_file, class2id_dict)

            # 이미지별 라벨(txt) 작성 & 이미지 복사
            for img_file in jpg_files:
                img_name = os.path.basename(img_file)
                
                # XML에 해당 이미지 정보가 있으면 가져오고, 없으면 빈 리스트
                boxes = annotations_dict.get(img_name, [])
                
                # 라벨 txt 경로
                txt_name = os.path.splitext(img_name)[0] + ".txt"
                txt_path = os.path.join(DATASET_DIR, "labels", subset, txt_name)
                
                # txt 생성
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for (class_id, xc, yc, w, h) in boxes:
                        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                
                # 이미지 복사
                dst_img_path = os.path.join(DATASET_DIR, "images", subset, img_name)
                shutil.copy2(img_file, dst_img_path)

                # 진행률 증가
                processed_count += 1
                progress = (processed_count / total_jpg_count) * 100
                # 콘솔에 진행률 표시 (같은 줄에서 갱신)
                print(f"[{processed_count}/{total_jpg_count}] ({progress:.2f}%)", end='\r')

    # (마지막에 줄바꿈)
    print()

    # ----------------------------------------------------
    # 5) dataset.yaml 파일 생성
    # ----------------------------------------------------
    # class2id_dict를 id 기준 정렬 -> class_names 리스트
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

    # 최종 완료 메시지
    print("----- 완료 -----")
    print(f"생성된 클래스 ID 매핑: {class2id_dict}")
    print(f"dataset.yaml 생성 위치: {yaml_path}")
    print(f"train/val 분류결과 파일: {split_info_path}")


if __name__ == "__main__":
    main()
