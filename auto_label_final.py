"""
Roboflow 자동 라벨링 - 최종 버전
SDK upload으로 이미지+YOLO어노테이션 동시 업로드
"""
import os
import sys
import shutil
from roboflow import Roboflow

API_KEY = "5XX326NDaVcMsUiVJ726"
WORKSPACE = "laptimev2log"
PROJECT = "v2log-weight-plates"
BASE_DIR = "C:/Dev/V2log-CV-Training/data/images/train"
TEMP = "C:/Dev/V2log-CV-Training/_temp"

CLASS_NAMES = ["plate_2.5kg", "plate_5kg", "plate_10kg", "plate_15kg", "plate_20kg"]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

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


def create_annotation(classes):
    if not classes:
        return ""
    lines = []
    if len(classes) == 1:
        lines.append(f"{CLASS_MAP[classes[0]]} 0.5 0.5 0.7 0.7")
    else:
        n = len(classes)
        for i, cls in enumerate(classes):
            sw = 0.8 / n
            cx = 0.1 + sw * (i + 0.5)
            lines.append(f"{CLASS_MAP[cls]} {cx:.3f} 0.500 {sw*0.9:.3f} 0.600")
    return "\n".join(lines)


def main():
    # stdout 버퍼링 비활성화
    sys.stdout.reconfigure(line_buffering=True)

    os.makedirs(TEMP, exist_ok=True)

    print("Roboflow 연결 중...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    print(f"프로젝트: {project.name}")

    total = 0
    done = 0
    errors = 0

    for folder_name, classes in FOLDER_TO_CLASSES.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[건너뜀] {folder_name}")
            continue

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        class_str = ", ".join(classes) if classes else "노이즈"
        print(f"\n--- {folder_name} ({len(images)}장) → [{class_str}] ---")

        ann_content = create_annotation(classes)

        for img_file in images:
            total += 1
            img_src = os.path.join(folder_path, img_file)

            # 영문 경로 임시 복사
            temp_img = os.path.join(TEMP, img_file)
            temp_ann = os.path.join(TEMP, os.path.splitext(img_file)[0] + ".txt")

            try:
                shutil.copy2(img_src, temp_img)
                with open(temp_ann, "w") as f:
                    f.write(ann_content)

                project.upload(
                    image_path=temp_img,
                    annotation_path=temp_ann,
                    split="train",
                )
                done += 1

                if done % 20 == 0:
                    print(f"  [{done}장 완료] ({total}장 중)")

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [에러] {img_file}: {e}")
            finally:
                for f in [temp_img, temp_ann]:
                    if os.path.exists(f):
                        os.remove(f)

    if os.path.exists(TEMP):
        shutil.rmtree(TEMP)

    print(f"\n{'='*40}")
    print(f"완료! 성공: {done} / 총: {total} / 에러: {errors}")


if __name__ == "__main__":
    main()
