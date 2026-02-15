"""
V2log Phase 2A - YOLO26 모델 변환 (.pt → .tflite)
===================================================

학습 완료 후 실행. 변환된 .tflite 파일을 V2log 앱에 넣으면 됨.

결과물: models/weight_plate/weights/best_float16.tflite
  → 복사 위치: C:\Dev\V2log\assets\models\weight_plate.tflite
"""

from ultralytics import YOLO

# Step 1: 학습된 모델 로드
model = YOLO('../models/weight_plate/weights/best.pt')

# Step 2: TFLite로 변환 (FP16 - 모바일 최적화)
model.export(
    format='tflite',
    half=True,        # FP16 (파일 크기 절반, 속도 향상)
    imgsz=640,        # 학습 시와 동일
)

print("\n변환 완료!")
print("파일: models/weight_plate/weights/best_float16.tflite")
print("\n다음 단계:")
print("1. best_float16.tflite → V2log/assets/models/weight_plate.tflite 복사")
print("2. V2log 프로젝트에서 Phase 2B 앱 통합 시작")
