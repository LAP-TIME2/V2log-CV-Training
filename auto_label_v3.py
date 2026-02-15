"""
Roboflow REST API 직접 호출로 이미지+어노테이션 업로드
"""
import os
import base64
import requests
import time

API_KEY = "5XX326NDaVcMsUiVJ726"
PROJECT_URL = "v2log-weight-plates"
UPLOAD_URL = f"https://api.roboflow.com/dataset/{PROJECT_URL}/upload"
ANNOTATE_URL = f"https://api.roboflow.com/dataset/{PROJECT_URL}/annotate"
BASE_DIR = "C:/Dev/V2log-CV-Training/data/images/train"

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


def create_yolo_annotation(classes):
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
            sw = 0.8 / n
            cx = 0.1 + sw * (i + 0.5)
            lines.append(f"{CLASS_MAP[cls]} {cx:.3f} 0.500 {sw*0.9:.3f} 0.600")
    return "\n".join(lines)


def upload_image(img_path, img_name):
    """이미지를 Roboflow에 업로드하고 image_id 반환"""
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    resp = requests.post(
        UPLOAD_URL,
        params={
            "api_key": API_KEY,
            "name": img_name,
            "split": "train",
            "overwrite": "true",
        },
        data=img_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return resp


def annotate_image(img_name, annotation_str, img_w=640, img_h=640):
    """YOLO 어노테이션을 Roboflow에 업로드"""
    resp = requests.post(
        f"{ANNOTATE_URL}/{img_name}",
        params={"api_key": API_KEY, "name": img_name, "overwrite": "true"},
        json={
            "annotationFile": annotation_str,
            "annotationFormat": "yolov8",
            "classes": CLASS_NAMES,
        },
    )
    return resp


def main():
    total = 0
    uploaded = 0
    annotated = 0
    errors = 0

    for folder_name, classes in FOLDER_TO_CLASSES.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[건너뜀] {folder_name}")
            continue

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        class_str = ", ".join(classes) if classes else "노이즈"
        print(f"\n=== {folder_name} ({len(images)}장) → [{class_str}] ===")

        for img_file in images:
            total += 1
            img_path = os.path.join(folder_path, img_file)

            try:
                # 1. 이미지 업로드
                resp = upload_image(img_path, img_file)
                if resp.status_code == 200:
                    uploaded += 1
                else:
                    # 이미 업로드된 이미지일 수 있음 - 어노테이션만 시도
                    pass

                # 2. 어노테이션 업로드
                ann = create_yolo_annotation(classes)
                if ann:  # 노이즈 제외
                    ann_resp = annotate_image(img_file, ann)
                    if ann_resp.status_code == 200:
                        annotated += 1
                    else:
                        if errors < 3:
                            print(f"  [어노테이션 에러] {img_file}: {ann_resp.status_code} {ann_resp.text[:100]}")
                        errors += 1

                if (uploaded + annotated) % 40 == 0:
                    print(f"  진행: {total}장 처리 중... (업로드:{uploaded} 어노테이션:{annotated})")

                # API 속도 제한 방지
                time.sleep(0.1)

            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [에러] {img_file}: {e}")

    print(f"\n{'='*40}")
    print(f"완료!")
    print(f"총: {total} | 업로드: {uploaded} | 어노테이션: {annotated} | 에러: {errors}")


if __name__ == "__main__":
    main()
