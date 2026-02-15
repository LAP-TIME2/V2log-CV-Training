# CLAUDE.md — V2log CV Training (Phase 2A)

> V2log 앱의 **무게 자동 감지 AI 모델**을 학습시키는 프로젝트.
> 앱 코드는 여기 없음 → 앱 개발은 `C:\Dev\V2log\` 참조.

---

## 프로젝트 관계

```
C:\Dev\V2log\                  ← Flutter 앱 (메인 프로젝트)
├── CLAUDE.md                  ← 앱 전체 가이드
├── CLAUDE-CV.md               ← CV 기능 개발 가이드
└── lib/                       ← 앱 코드

C:\Dev\V2log-CV-Training\      ← ⭐ 이 프로젝트 (AI 모델 학습)
├── CLAUDE.md                  ← 이 파일
├── data/                      ← 사진 + 라벨
├── scripts/                   ← 학습/변환 코드
├── models/                    ← 학습 결과물
└── notebooks/                 ← Colab 가이드
```

### 전체 CV 로드맵에서의 위치

```
Phase 1  (횟수 카운팅)     ✅ 완료 — V2log 앱에 구현됨
Phase 1.5 (스마트 무게)     ⏳ 미착수
Phase 2A (모델 학습)        ⭐ 이 프로젝트 — Python/Colab
Phase 2B (앱 통합)          ⏳ Phase 2A 완료 후 — V2log 앱에 추가
Phase 3  (미래 기술)        ⏳ 미착수
```

---

## 이 프로젝트의 목표

**입력**: 헬스장 플레이트 사진 3,000~5,000장
**출력**: `best.tflite` 파일 1개 (< 10MB)
**용도**: V2log 앱에서 카메라로 바벨을 비추면 무게를 자동 감지

---

## 기술 스택

| 용도 | 기술 | 비고 |
|------|------|------|
| 객체 인식 모델 | **YOLO26-N** | 모바일 최적화, 가장 작고 빠른 버전 |
| 학습 환경 | **Google Colab** (무료 T4 GPU) | 로컬 GPU 불필요 |
| 라벨링 도구 | **Roboflow** (웹) | 무료 10,000장, AI 보조 라벨링 |
| 프레임워크 | **Ultralytics** (Python) | YOLO 공식 라이브러리 |
| 최종 모델 형식 | **TFLite** (FP16) | Flutter tflite_flutter로 실행 |

---

## 감지 대상 (9개 클래스)

| # | 클래스명 | 실제 무게 | IWF 표준 색상 |
|---|---------|----------|--------------|
| 0 | plate_25kg | 25kg | 빨강 |
| 1 | plate_20kg | 20kg | 파랑 |
| 2 | plate_15kg | 15kg | 노랑 |
| 3 | plate_10kg | 10kg | 초록 |
| 4 | plate_5kg | 5kg | 흰색 |
| 5 | plate_2.5kg | 2.5kg | 검정 (작은) |
| 6 | plate_1.25kg | 1.25kg | 검정 (가장 작은) |
| 7 | barbell | — | 플레이트가 끼워진 바벨 전체 |
| 8 | empty_barbell | 20kg (표준) | 빈 바벨 |

---

## 작업 워크플로우

### Step 1: 데이터 수집 (헬스장 촬영)
- **목표**: 3,000~5,000장 (클래스당 300~500장)
- **장소**: 헬스장 3~5곳 (조명/기구 다양성 확보)
- **촬영 가이드**: `scripts/photo_guide.md`
- **핵심**: 다양한 각도/조명/거리 > 깔끔한 사진

### Step 2: 라벨링 (Roboflow)
- **도구**: https://roboflow.com (웹 브라우저)
- **방법**: 사진 업로드 → 바운딩 박스 그리기 → 클래스 지정
- **효율화**: 500장 수동 → 임시 모델 학습 → 나머지 자동 라벨링 + 수정
- **데이터 증강**: Roboflow 자동 (밝기/회전/노이즈) → ×5~10배
- **최종 데이터**: 8,000~10,000장 (증강 포함)
- **분할 비율**: train 70% / val 20% / test 10%

### Step 3: 모델 학습 (Google Colab)
- **환경 셋업**: `notebooks/colab_setup.md`
- **학습 코드**: `scripts/train.py` (핵심 3줄)
- **GPU**: T4 (Colab 무료) 또는 A100 (Colab Pro)
- **예상 시간**: 5,000장 기준 2~4시간
- **목표 정확도**: mAP50 ≥ 80%

### Step 4: 검증
- **검증 코드**: `scripts/validate.py`
- **판정 기준**:
  - mAP50 ≥ 80% → 통과 (앱에 넣을 수 있음)
  - mAP50 60~80% → 데이터 추가 또는 하이퍼파라미터 조정
  - mAP50 < 60% → 데이터 품질/양 근본적 개선 필요

### Step 5: 변환 및 전달
- **변환 코드**: `scripts/export.py` (핵심 1줄)
- **결과물**: `models/weight_plate/weights/best_float16.tflite`
- **전달**: → `C:\Dev\V2log\assets\models\weight_plate.tflite`로 복사
- **이후**: V2log 프로젝트에서 Phase 2B (앱 통합) 시작

---

## 폴더 구조

```
V2log-CV-Training/
├── CLAUDE.md                    ← 이 파일 (프로젝트 가이드)
├── data.yaml                    ← YOLO 학습 설정 (클래스 정의)
├── data/
│   ├── images/
│   │   ├── train/               ← 학습용 사진 (70%)
│   │   ├── val/                 ← 검증용 사진 (20%)
│   │   └── test/                ← 테스트용 사진 (10%)
│   └── labels/
│       ├── train/               ← Roboflow YOLO 형식 라벨
│       ├── val/
│       └── test/
├── scripts/
│   ├── train.py                 ← 학습 실행 (3줄 핵심)
│   ├── export.py                ← .pt → .tflite 변환
│   ├── validate.py              ← 정확도 검증
│   └── photo_guide.md           ← 촬영 가이드
├── notebooks/
│   └── colab_setup.md           ← Colab 환경 셋업 가이드
└── models/                      ← 학습 결과물 저장
    └── weight_plate/
        └── weights/
            ├── best.pt          ← 학습된 모델 (PyTorch)
            └── best_float16.tflite  ← 변환된 모델 (모바일용)
```

---

## Phase 1 (횟수 카운팅) 요약 — 이미 완료된 내용

> Phase 2A 작업할 때 Phase 1 맥락이 필요할 수 있으므로 요약.

- **기술**: MediaPipe BlazePose (33개 관절) + 관절 각도 Peak/Valley
- **알고리즘**: One Euro Filter + Velocity Gate + 확인 시스템 (v5.2)
- **정확도**: 88~95% (헬스장 환경)
- **지원 운동**: 10개 (바이셉 컬, 스쿼트, 벤치프레스, 숄더프레스, 랫풀다운 등)
- **핵심 파일**: `V2log/lib/data/services/rep_counter_service.dart` (655줄)
- **수정노트**: `V2log/docs/CV_수정노트.md`

---

## 개발 진행 상황

### Step 1: 데이터 수집 — **미착수**
- [ ] 헬스장 1곳 촬영 (파일럿)
- [ ] 헬스장 2~3곳 추가 촬영
- [ ] 총 3,000장 이상 확보
- [ ] 클래스 분포 확인 (편향 없는지)

### Step 2: 라벨링 — **미착수**
- [ ] Roboflow 계정 생성 + 프로젝트 생성
- [ ] 500장 수동 라벨링
- [ ] 임시 모델로 자동 라벨링 + 수정
- [ ] 데이터 증강 적용
- [ ] data/ 폴더로 export (YOLO 형식)

### Step 3: 모델 학습 — **미착수**
- [ ] Colab 환경 셋업
- [ ] YOLO26-N 학습 실행
- [ ] 학습 로그 확인 (loss 수렴, 과적합 체크)

### Step 4: 검증 — **미착수**
- [ ] 테스트 데이터 검증 (mAP50 ≥ 80% 목표)
- [ ] 클래스별 정확도 확인 (약한 클래스 파악)
- [ ] 필요시 데이터 추가 + 재학습

### Step 5: 변환 및 전달 — **미착수**
- [ ] .pt → .tflite 변환
- [ ] 모델 파일 크기 확인 (< 10MB)
- [ ] V2log 앱으로 전달 → Phase 2B 시작

---

## B2B Fine-tuning 전략 (Phase 2B 이후)

- 헬스장 관리자가 앱에서 플레이트 촬영 200~300장
- 사진마다 터치로 무게 선택 (라벨링)
- 서버에서 기본 모델 + 관리자 데이터 → 맞춤 모델 자동 생성
- 해당 헬스장에서 90~95% 정확도 기대

---

## 비용

| 항목 | 비용 |
|------|------|
| Roboflow | 무료 (10,000장까지) |
| Google Colab | 무료 (T4 GPU) |
| Ultralytics | 무료 (오픈소스) |
| **총합** | **0원** (시간만 투자) |

---

## 참고 문서 (V2log 프로젝트)

- **CV 개발 가이드**: `C:\Dev\V2log\CLAUDE-CV.md`
- **기술 조사 보고서**: `C:\Dev\V2log\docs\reference\CV_무게측정_횟수카운팅_최신기술_보고서_2026.md`
- **사업계획서**: `C:\Dev\V2log\docs\reference\CV_피벗_사업계획서_Fica_2026.md`
- **CV 수정노트**: `C:\Dev\V2log\docs\CV_수정노트.md`
- **앱 전체 가이드**: `C:\Dev\V2log\CLAUDE.md`
