import os
import xml.etree.ElementTree as ET
import csv

# 기준 디렉토리 설정
base_dir = "data/15.인도보행영상/바운딩박스"
output_csv = "output.csv"

# CSV 파일 생성 및 헤더 작성
with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # CSV 헤더 작성
    writer.writerow(["image_id", "image_path", "label", "occluded", "xtl", "ytl", "xbr", "ybr", "z_order"])

    # 모든 Bbox 디렉토리 탐색
    for bbox_dir in os.listdir(base_dir):
        bbox_path = os.path.join(base_dir, bbox_dir)

        # 디렉토리가 맞는지 확인
        if os.path.isdir(bbox_path) and bbox_dir.startswith("Bbox_"):
            # XML 파일 찾기
            for file in os.listdir(bbox_path):
                if file.endswith(".xml"):
                    xml_path = os.path.join(bbox_path, file)

                    # XML 파일 파싱
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    # XML의 이미지 데이터 읽기
                    for image in root.findall("image"):
                        image_id = image.get("id")
                        image_name = image.get("name")
                        image_path = os.path.join(bbox_dir, image_name)

                        # 각 box 태그에서 데이터 추출
                        for box in image.findall("box"):
                            label = box.get("label")
                            occluded = box.get("occluded")
                            xtl = box.get("xtl")
                            ytl = box.get("ytl")
                            xbr = box.get("xbr")
                            ybr = box.get("ybr")
                            z_order = box.get("z_order")

                            # CSV 파일에 데이터 작성
                            writer.writerow([image_id, image_path, label, occluded, xtl, ytl, xbr, ybr, z_order])

print(f"CSV 파일이 생성되었습니다: {output_csv}")
