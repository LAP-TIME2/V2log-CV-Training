# Google Colab YOLO26-N 학습 초상세 가이드

> 이 가이드대로 하면 클릭 하나하나 따라할 수 있습니다.
> 예상 소요 시간: 업로드 10분 + 학습 1~2시간 + 변환 5분

---

## PART 1: Google Drive에 데이터셋 업로드

### Step 1: Google Drive 열기
- 브라우저에서 https://drive.google.com 접속
- → 화면: Google Drive 메인 화면 (내 드라이브)
- → 이미 로그인되어 있으면 바로 보임. 안 되어 있으면 Google 계정 로그인

### Step 2: 폴더 만들기
- → 왼쪽 상단 "+ 새로 만들기" 버튼 클릭
- → "새 폴더" 클릭
- → 폴더 이름: `V2log-CV-Training` 입력
- → "만들기" 클릭
- → 결과: 내 드라이브에 `V2log-CV-Training` 폴더 생성됨

### Step 3: 폴더 들어가기
- → 방금 만든 `V2log-CV-Training` 폴더 더블클릭
- → 빈 폴더 내부로 들어감

### Step 4: 데이터셋 ZIP 업로드 (압축 풀기 전 원본 ZIP!)
- ⚠️ **압축 푼 폴더가 아니라, 원본 ZIP 파일을 업로드하세요!**
  - Roboflow에서 받은 `.zip` 파일 (예: `v2log-weight-plates-2.zip`)
  - 이유: 폴더째 업로드하면 수백 개 파일이 하나씩 올라가서 매우 느림. ZIP 1개가 훨씬 빠름
- → 화면 빈 공간에 ZIP 파일을 **드래그 앤 드롭**
- → 또는: 왼쪽 상단 "+ 새로 만들기" → "파일 업로드" → ZIP 선택
- → 오른쪽 하단에 업로드 진행바 나옴
- → 기다리기 (파일 크기에 따라 1~10분)
- → 결과: Drive에 ZIP 파일 1개 올라감

---

## PART 2: Google Colab 열기 + GPU 설정

### Step 5: Colab 접속
- 브라우저 새 탭에서 https://colab.research.google.com 접속
- → 화면: "노트 열기" 팝업이 뜰 수 있음 → 우측 하단 "취소" 또는 "새 노트" 클릭
- → 또는 팝업 없이 바로 빈 노트북이 열릴 수도 있음

### Step 6: 새 노트북 만들기
- → 왼쪽 상단 "파일" 메뉴 클릭
- → "새 노트" 클릭
- → 결과: 빈 코드 셀 1개가 있는 새 노트북 열림
- → 노트북 이름(상단 "Untitled0.ipynb")을 클릭해서 `V2log_YOLO26_Training` 으로 변경

### Step 7: GPU 런타임 설정 (⚠️ 필수!)
- → 상단 메뉴에서 "런타임" 클릭
- → "런타임 유형 변경" 클릭
- → 팝업이 뜸:
  - **하드웨어 가속기**: 드롭다운에서 `T4 GPU` 선택
    - ✅ `T4 GPU` ← 이거!
    - ❌ `None` (CPU만 = 학습 불가능하게 느림)
    - ❌ `A100` (유료 Colab Pro 전용)
    - ❌ `TPU` (YOLO 호환 안 됨)
  - 나머지 설정은 그대로 두기
- → "저장" 클릭
- → 결과: 우측 상단에 "T4" 또는 GPU 아이콘 표시됨

### Step 8: GPU 연결 확인
- → 우측 상단 "연결" 버튼 클릭 (이미 연결되어 있으면 "RAM/디스크" 표시)
- → 잠시 기다리면 연결됨
- → "연결됨" 표시 + RAM/디스크 게이지 보이면 OK

---

## PART 3: 학습 코드 실행 (셀 5개)

> 아래 코드를 **한 셀씩** 복사해서 Colab에 붙여넣고 실행합니다.
> 셀 실행: 셀 왼쪽의 ▶ 버튼 클릭, 또는 Ctrl+Enter

### 셀 1: Google Drive 연결
```python
# Google Drive 연결 (데이터셋이 여기 있음)
from google.colab import drive
drive.mount('/content/drive')
```
- → 실행하면 "Google Drive에 액세스 허용" 팝업 뜸
- → "Google Drive에 연결" 클릭
- → Google 계정 선택 (Drive에 업로드한 계정과 동일해야 함!)
- → "허용" 클릭
- → 결과: `Mounted at /content/drive` 메시지 나오면 성공

### 셀 2: 데이터셋 압축 해제 + 폴더 구조 확인
```python
import zipfile
import os

# ZIP 파일 경로 (⚠️ 파일명을 실제 이름으로 수정!)
zip_path = '/content/drive/MyDrive/V2log-CV-Training/v2log-weight-plates-2.zip'

# 압축 해제
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')

# 폴더 구조 확인
for root, dirs, files in os.walk('/content/dataset'):
    level = root.replace('/content/dataset', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:  # 2단계까지만 표시
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... 외 {len(files)-5}개')
```
- → ⚠️ `zip_path`의 파일명을 **실제 ZIP 파일명**으로 수정!
  - Drive에서 ZIP 파일명 확인 → 그대로 입력
- → 실행하면 폴더 구조 출력됨
- → train/, test/, valid/ 폴더가 보이면 OK

### 셀 3: data.yaml 경로 수정 + Ultralytics 설치
```python
# Ultralytics (YOLO) 설치
!pip install ultralytics -q

# data.yaml 읽기 + 경로 수정
import yaml

yaml_path = '/content/dataset/data.yaml'

with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# Roboflow 경로 → Colab 절대 경로로 수정
data['path'] = '/content/dataset'
data['train'] = 'train/images'
data['val'] = 'valid/images'
data['test'] = 'test/images'

with open(yaml_path, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print("data.yaml 수정 완료!")
print(f"클래스 수: {data['nc']}")
print(f"클래스: {data['names']}")
```
- → 실행하면 pip 설치 로그 쭉 나오고 마지막에 클래스 정보 출력
- → 클래스 수와 이름이 맞는지 확인

### 셀 4: YOLO26-N 학습 시작 (⏱️ 30분~2시간)
```python
from ultralytics import YOLO

# YOLO26-N 모델 로드 (2026년 1월 출시, 최신 모델)
# 첫 실행 시 자동 다운로드됨 (~6MB)
model = YOLO('yolo26n.pt')

# 학습 시작!
results = model.train(
    data='/content/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
    save=True,
    project='/content/models',
    name='weight_plate',
)

# 결과 출력
print(f"\n{'='*50}")
print(f"학습 완료!")
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"{'='*50}")
```
- → 실행하면 모델 다운로드 → 학습 시작
- → 화면에 epoch별 진행 상황 표시 (epoch 1/100, 2/100, ...)
- → ⏱️ 데이터 양에 따라 30분~2시간 소요
- → ⚠️ **학습 중 브라우저 탭 닫지 마세요!** (Colab 세션 끊김)
- → ⚠️ 화면 보호기/절전 모드도 주의 (세션 끊길 수 있음)
- → 학습 끝나면 mAP50 숫자 나옴:
  - **mAP50 ≥ 80%** → 🎉 성공! Step 5로
  - **mAP50 60~80%** → 나쁘진 않음, 일단 변환 후 테스트
  - **mAP50 < 60%** → 데이터 추가 필요, 나한테 말해주세요

### 셀 5: TFLite 변환 + Google Drive에 저장
```python
from ultralytics import YOLO
import shutil

# 학습된 모델 로드
model = YOLO('/content/models/weight_plate/weights/best.pt')

# TFLite로 변환
model.export(format='tflite', half=True, imgsz=640)

# Google Drive에 결과물 복사
src = '/content/models/weight_plate/weights/'
dst = '/content/drive/MyDrive/V2log-CV-Training/results/'
os.makedirs(dst, exist_ok=True)

# 중요 파일들 복사
for f in ['best.pt', 'best_float16.tflite']:
    src_file = os.path.join(src, f)
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst)
        print(f"✅ 저장됨: {dst}{f}")

# 학습 결과 그래프도 복사
results_dir = '/content/models/weight_plate/'
for f in os.listdir(results_dir):
    if f.endswith('.png') or f.endswith('.csv'):
        shutil.copy2(os.path.join(results_dir, f), dst)

print(f"\n모든 결과물이 Google Drive에 저장되었습니다!")
print(f"위치: Drive > V2log-CV-Training > results")
```
- → 변환 + Drive 저장까지 자동
- → 결과물이 Google Drive `V2log-CV-Training/results/` 에 저장됨
- → 여기서 `best_float16.tflite` 파일을 나중에 V2log 앱에 넣을 거예요

---

## PART 4: 결과물 다운로드

### Step 9: Google Drive에서 다운로드
- → Google Drive 열기 (https://drive.google.com)
- → `V2log-CV-Training` 폴더 → `results` 폴더 들어가기
- → 파일 목록:
  - `best.pt` — 학습된 모델 (PyTorch, PC용)
  - `best_float16.tflite` — 변환된 모델 (모바일 앱용) ← **이게 최종 결과물!**
  - `results.png` — 학습 그래프
  - `confusion_matrix.png` — 혼동 행렬 (클래스별 정확도)
- → `best_float16.tflite` 우클릭 → "다운로드"

---

## 문제 해결

### "GPU를 사용할 수 없습니다" 에러
- → Colab 무료 GPU 할당량 초과
- → 해결: 몇 시간 기다렸다가 다시 시도, 또는 다른 Google 계정 사용

### 학습 중 "세션이 끊겼습니다"
- → 브라우저 탭을 오래 비활성으로 두면 발생
- → 해결: 탭을 가끔 클릭해주기, 또는 Colab 설정에서 "유휴 시간 초과" 확인

### "yolo26n.pt not found"
- → YOLO26 모델이 아직 ultralytics에 안 올라온 경우
- → 해결: 셀 4의 코드에서 `yolo11n.pt` 사용 (이미 대비해놨음)

### mAP50이 너무 낮을 때 (< 60%)
- → 데이터가 부족하거나, 라벨링이 부정확할 수 있음
- → 해결: 나한테 결과 스크린샷 보여주세요, 같이 분석합니다

---

## 체크리스트

- [ ] Google Drive에 ZIP 업로드
- [ ] Colab 새 노트북 생성
- [ ] GPU 런타임 T4 설정
- [ ] 셀 1: Drive 연결
- [ ] 셀 2: 데이터셋 압축 해제
- [ ] 셀 3: data.yaml 수정 + ultralytics 설치
- [ ] 셀 4: YOLO 학습 실행 (30분~2시간 대기)
- [ ] 셀 5: TFLite 변환 + Drive 저장
- [ ] Drive에서 best_float16.tflite 다운로드
