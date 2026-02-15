"""
V2log Phase 2A - YOLO26 Weight Plate Detection Training
========================================================

사용법 (Google Colab):
1. 이 파일을 Colab에 업로드
2. Roboflow에서 라벨링한 데이터를 data/ 폴더에 다운로드
3. 아래 코드 실행 (GPU 런타임 필수)

예상 시간: 5,000장 기준 2~4시간 (T4 GPU)
비용: 무료 (Colab Free)
"""

from ultralytics import YOLO

# Step 1: 기본 모델 로드 (YOLO26 nano - 모바일 최적화)
model = YOLO('yolo26n.pt')

# Step 2: 학습 시작
results = model.train(
    data='../data.yaml',    # 데이터 설정
    epochs=100,             # 학습 반복 횟수
    imgsz=640,              # 이미지 크기
    batch=16,               # 배치 크기 (Colab Free: 16, Pro: 32)
    device=0,               # GPU 사용 (0 = 첫 번째 GPU)
    patience=20,            # 20 에폭 동안 개선 없으면 조기 종료
    save=True,              # 체크포인트 저장
    project='../models',    # 결과 저장 경로
    name='weight_plate',    # 실험 이름
)

# Step 3: 결과 확인
print(f"\n{'='*50}")
print(f"학습 완료!")
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"모델 저장: models/weight_plate/weights/best.pt")
print(f"{'='*50}")
