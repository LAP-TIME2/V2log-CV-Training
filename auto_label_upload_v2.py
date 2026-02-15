"""
Roboflow 자동 라벨링 업로드 스크립트 v2
- 한글 경로 문제 해결: 임시 영문 경로로 복사 후 업로드
- 폴더 이름 기반으로 클래스 자동 지정
"""

import os
import shutil
import tempfile
from roboflow import Roboflow

# === 설정 ===
API_KEY = "5XX326NDaVcMsUiVJ726"
WORKSPACE = "laptimev2log"
PROJECT = "v2log-weight-plates"
BASE_DIR = "C:/Dev/V2log-CV-Training/data/images/train"

# 임시 작업 디렉토리 (영문 경로)
TEMP_DIR = "C:/Dev/V2log-CV-Training/_temp_upload"

# 클래스 ID 매핑 (YOLO format)
CLASS_NAMES = ["plate_2.5kg", "plate_5kg", "plate_10kg", "plate_15kg", "plate_20kg"]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# 폴더 이름 → 클래스 목록 매핑
FOLDER_TO_CLASSES = {
    "2.5kg": ["plate_2.5kg"],
    "2.5kg 여러장": ["plate_2.5kg"],
    "5kg": ["plate_5kg"],
    "5kg 여러장": ["plate_5kg"],
    "10gk": ["plate_10kg"],
    "10kg 여러장": ["plate_10kg"],
    "15kg": ["plate_15kg"],
    "15kg 여러장": ["plate_15kg"],
    "20kg": ["plate_20kg"],
    "20gk 여러장": ["plate_20kg"],
    "20kg 10kg": ["plate_20kg", "plate_10kg"],
    "20kg 10kg 5kg": ["plate_20kg", "plate_10kg", "plate_5kg"],
    "20kg 10kg 5 kg 2.5kg": ["plate_20kg", "plate_10kg", "plate_5kg", "plate_2.5kg"],
    "20kg 15kg": ["plate_20kg", "plate_15kg"],
    "20kg 15kg 10kg": ["plate_20kg", "plate_15kg", "plate_10kg"],
    "20kg 15kg 10kg 5kg": ["plate_20kg", "plate_15kg", "plate_10kg", "plate_5kg"],
    "20kg 15kg 10kg 5kg 2.5kg": ["plate_20kg", "plate_15kg", "plate_10kg", "plate_5kg", "plate_2.5kg"],
    "노이즈": [],
}


def create_yolo_annotation(classes):
    """YOLO 형식 어노테이션 생성"""
    if not classes:
        return ""

    lines = []
    if len(classes) == 1:
        class_id = CLASS_MAP[classes[0]]
        lines.append(f"{class_id} 0.5 0.5 0.7 0.7")
    else:
        n = len(classes)
        for i, cls in enumerate(classes):
            class_id = CLASS_MAP[cls]
            segment_width = 0.8 / n
            cx = 0.1 + segment_width * (i + 0.5)
            cy = 0.5
            w = segment_width * 0.9
            h = 0.6
            lines.append(f"{class_id} {cx:.3f} {cy:.3f} {w:.3f} {h:.3f}")

    return "\n".join(lines)


def main():
    # 임시 디렉토리 준비
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Roboflow 연결
    print("Roboflow 연결 중...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    print(f"프로젝트: {project.name}")

    total = 0
    uploaded = 0
    errors = 0

    for folder_name, classes in FOLDER_TO_CLASSES.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[건너뜀] {folder_name}")
            continue

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

        class_str = ", ".join(classes) if classes else "노이즈"
        print(f"\n=== {folder_name} ({len(images)}장) → [{class_str}] ===")

        for img_file in images:
            total += 1
            img_src = os.path.join(folder_path, img_file)

            # 영문 임시 경로로 복사
            temp_img = os.path.join(TEMP_DIR, img_file)
            temp_ann = os.path.join(TEMP_DIR, os.path.splitext(img_file)[0] + ".txt")

            try:
                shutil.copy2(img_src, temp_img)

                # 어노테이션 파일 생성
                ann_content = create_yolo_annotation(classes)
                with open(temp_ann, "w") as f:
                    f.write(ann_content)

                # Roboflow 업로드
                project.upload(
                    image_path=temp_img,
                    annotation_path=temp_ann,
                    split="train",
                )
                uploaded += 1

                if uploaded % 20 == 0:
                    print(f"  [{uploaded}/{total}] 업로드 중...")

            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [에러] {img_file}: {e}")
            finally:
                # 임시 파일 정리
                if os.path.exists(temp_img):
                    os.remove(temp_img)
                if os.path.exists(temp_ann):
                    os.remove(temp_ann)

    # 정리
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    print(f"\n{'='*40}")
    print(f"완료!")
    print(f"총: {total} | 성공: {uploaded} | 에러: {errors}")


if __name__ == "__main__":
    main()
