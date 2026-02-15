# Google Colab 셋업 가이드

## 1. Colab 접속
- https://colab.research.google.com
- 구글 계정 로그인
- 새 노트북 생성

## 2. GPU 런타임 설정
- 메뉴: 런타임 → 런타임 유형 변경 → T4 GPU 선택
- 무료 계정: T4 GPU (충분함)

## 3. 첫 번째 셀: 패키지 설치
```python
!pip install ultralytics roboflow
```

## 4. 두 번째 셀: Roboflow에서 데이터 다운로드
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")  # Roboflow에서 발급
project = rf.workspace().project("v2log-weight-plates")
dataset = project.version(2).download("yolov8")
```

## 5. 세 번째 셀: 학습 실행
```python
from ultralytics import YOLO

model = YOLO('yolo26n.pt')
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16)
```

## 6. 네 번째 셀: TFLite 변환
```python
model = YOLO('runs/detect/train/weights/best.pt')
model.export(format='tflite', half=True)
```

## 7. 결과 다운로드
- `runs/detect/train/weights/best_float16.tflite` 파일 다운로드
- V2log/assets/models/ 에 복사

## 예상 시간
- 5,000장, 100 에폭: 약 2~4시간
- 10,000장, 100 에폭: 약 4~6시간
