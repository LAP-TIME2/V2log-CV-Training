"""
Roboflow 자동 라벨링 업로드 스크립트
- 폴더 이름 기반으로 클래스 자동 지정
- 이미지 중앙에 기본 바운딩 박스 생성
- Roboflow API로 업로드
"""

import os
import json
import glob
from PIL import Image
from roboflow import Roboflow

# === 설정 ===
API_KEY = "5XX326NDaVcMsUiVJ726"
WORKSPACE = "laptimev2log"
PROJECT = "v2log-weight-plates"
BASE_DIR = "C:/Dev/V2log-CV-Training/data/images/train"

# 클래스 ID 매핑 (YOLO format)
CLASS_MAP = {
    "plate_2.5kg": 0,
    "plate_5kg": 1,
    "plate_10kg": 2,
    "plate_15kg": 3,
    "plate_20kg": 4,
}

# 폴더 이름 → 클래스 목록 매핑
FOLDER_TO_CLASSES = {
    "2.5kg": ["plate_2.5kg"],
    "2.5kg 여러장": ["plate_2.5kg"],
    "5kg": ["plate_5kg"],
    "5kg 여러장": ["plate_5kg"],
    "10gk": ["plate_10kg"],  # 10kg 오타
    "10kg 여러장": ["plate_10kg"],
    "15kg": ["plate_15kg"],
    "15kg 여러장": ["plate_15kg"],
    "20kg": ["plate_20kg"],
    "20gk 여러장": ["plate_20kg"],  # 20kg 오타
    "20kg 10kg": ["plate_20kg", "plate_10kg"],
    "20kg 10kg 5kg": ["plate_20kg", "plate_10kg", "plate_5kg"],
    "20kg 10kg 5 kg 2.5kg": ["plate_20kg", "plate_10kg", "plate_5kg", "plate_2.5kg"],
    "20kg 15kg": ["plate_20kg", "plate_15kg"],
    "20kg 15kg 10kg": ["plate_20kg", "plate_15kg", "plate_10kg"],
    "20kg 15kg 10kg 5kg": ["plate_20kg", "plate_15kg", "plate_10kg", "plate_5kg"],
    "20kg 15kg 10kg 5kg 2.5kg": ["plate_20kg", "plate_15kg", "plate_10kg", "plate_5kg", "plate_2.5kg"],
    "노이즈": [],  # 노이즈 = 플레이트 없음
}


def create_yolo_annotation(classes, num_objects=None):
    """YOLO 형식 어노테이션 문자열 생성

    단일 클래스: 이미지 중앙에 큰 박스 1개
    복수 클래스: 바벨 기준으로 좌우/상하 배치 (바깥쪽이 큰 플레이트)
    """
    if not classes:
        return ""  # 노이즈: 빈 파일

    lines = []

    if len(classes) == 1:
        # 단일 클래스: 중앙에 큰 박스
        class_id = CLASS_MAP[classes[0]]
        lines.append(f"{class_id} 0.5 0.5 0.7 0.7")
    else:
        # 복수 클래스: 바벨에 끼워진 플레이트들
        # 큰 플레이트가 바깥, 작은 플레이트가 안쪽
        # 가로 배치 가정 (좌→우: 큰→작은)
        n = len(classes)
        for i, cls in enumerate(classes):
            class_id = CLASS_MAP[cls]
            # 균등 분할: 각 플레이트가 이미지의 일부를 차지
            segment_width = 0.8 / n
            cx = 0.1 + segment_width * (i + 0.5)
            cy = 0.5
            w = segment_width * 0.9
            h = 0.6
            lines.append(f"{class_id} {cx:.3f} {cy:.3f} {w:.3f} {h:.3f}")

    return "\n".join(lines)


def main():
    # Roboflow 연결
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    print(f"프로젝트 연결: {project.name}")
    print(f"기존 이미지 수: {project.annotation}")

    # 임시 디렉토리: 어노테이션 파일 저장용
    ann_dir = "C:/Dev/V2log-CV-Training/data/annotations_temp"
    os.makedirs(ann_dir, exist_ok=True)

    total = 0
    uploaded = 0
    errors = 0

    for folder_name, classes in FOLDER_TO_CLASSES.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[건너뜀] 폴더 없음: {folder_name}")
            continue

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

        print(f"\n--- {folder_name} ({len(images)}장) → {classes or '노이즈(빈 라벨)'} ---")

        for img_file in images:
            total += 1
            img_path = os.path.join(folder_path, img_file)

            # YOLO 어노테이션 파일 생성
            ann_content = create_yolo_annotation(classes)
            ann_file = os.path.splitext(img_file)[0] + ".txt"
            ann_path = os.path.join(ann_dir, ann_file)

            with open(ann_path, "w") as f:
                f.write(ann_content)

            try:
                # Roboflow에 이미지 + 어노테이션 업로드
                project.upload(
                    image_path=img_path,
                    annotation_path=ann_path,
                    split="train",
                    tag=folder_name,  # 폴더 이름을 태그로
                )
                uploaded += 1
                if uploaded % 50 == 0:
                    print(f"  진행: {uploaded}/{total}")
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [에러] {img_file}: {e}")
                elif errors == 6:
                    print("  ... 추가 에러 생략 ...")

    print(f"\n=== 완료 ===")
    print(f"총 이미지: {total}")
    print(f"업로드 성공: {uploaded}")
    print(f"에러: {errors}")


if __name__ == "__main__":
    main()
