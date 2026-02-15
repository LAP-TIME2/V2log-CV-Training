"""
V2log Phase 2A - 학습된 모델 검증
=================================

학습 후 테스트 데이터로 정확도를 확인하는 스크립트.
mAP50 80% 이상이면 앱에 넣을 수 있는 수준.
"""

from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO('../models/weight_plate/weights/best.pt')

# 테스트 데이터로 검증
results = model.val(
    data='../data.yaml',
    split='test',       # test 세트로 검증
    imgsz=640,
    batch=16,
)

print(f"\n{'='*50}")
print(f"검증 결과")
print(f"{'='*50}")
print(f"mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"\n클래스별 정확도:")

class_names = [
    'plate_25kg', 'plate_20kg', 'plate_15kg', 'plate_10kg',
    'plate_5kg', 'plate_2.5kg', 'plate_1.25kg', 'barbell', 'empty_barbell'
]

for i, name in enumerate(class_names):
    ap = results.results_dict.get(f'metrics/mAP50(B)', 'N/A')
    print(f"  {name}: {ap}")

print(f"\n판정: ", end="")
mAP = results.results_dict.get('metrics/mAP50(B)', 0)
if mAP >= 0.8:
    print(f"통과! (mAP50 {mAP:.1%} >= 80%)")
elif mAP >= 0.6:
    print(f"개선 필요 (mAP50 {mAP:.1%}, 목표 80%)")
else:
    print(f"데이터 추가 필요 (mAP50 {mAP:.1%})")
