# 🧠 Korean Emotion Classification (분노·불안·슬픔·평온·당황·기쁨)

## 📌 프로젝트 개요
한국어 대화 데이터를 기반으로 6개 감정(분노, 불안, 슬픔, 평온, 당황, 기쁨)을 분류하는 모델입니다.  
의사라벨링(pseudo-labeling) → 대규모 재학습 → 최종 파인튜닝의 3단계 파이프라인을 통해 안정성과 일반화를 모두 확보했습니다.  

본 모델은 Hugging Face Hub(@Seonghaa)에 업로드되어 있으며, 파이프라인 API로 바로 추론에 활용할 수 있습니다.

---

## ⚙️ 모델 학습 프로세스
1. **Step 1. KcELECTRA(Simple Training)**  
   - 베이스모델: `beomi/KcELECTRA-base-v2022`  
   - Label Smoothing=0.05, Gradient Clipping=1.0, EarlyStopping(patience=2)  
   - 클래스 불균형 보정: `class weights` 적용  
   - 결과: Macro-F1 기준 best 모델 저장  

2. **Step 2. 대규모 의사라벨링 (약 24만 문장)**  
   - Step 1에서 학습한 모델로 전체 raw 데이터셋을 재분류  
   - `pred_emotion`, `pred_conf`, `pred_margin` 추가  
   - 신뢰도/마진 필터링(`conf≥0.75`, `margin≥0.20`)을 통해 high-confidence 샘플 확보  

3. **Step 3. 최종 파인튜닝 (KLUE RoBERTa)**  
   - 베이스모델: `klue/roberta-base` (장문 및 문맥 이해 강점)  
   - 학습 데이터: Step 2 의사라벨링 결과 활용  
   - Label Smoothing=0.05, EarlyStopping(patience=2), bf16 mixed precision (A100)  
   - 평가 지표: Macro-F1, Weighted-F1, Accuracy  

---

## 📊 성능 지표 (골든셋 기준)
| 감정   | Precision | Recall  | F1-score |
|--------|-----------|---------|----------|
| 기쁨   | 0.9857    | 0.9886  | **0.9872** |
| 당황   | 0.9607    | 0.9668  | **0.9652** |
| 분노   | 0.9801    | 0.9788  | **0.9795** |
| 불안   | 0.9864    | 0.9848  | **0.9856** |
| 평온   | 0.9782    | 0.9750  | **0.9766** |
| 슬픔   | 0.9837    | 0.9854  | **0.9845** |
| **전체 평균** | - | - | **0.9831 (Accuracy)** |

---

## 📂 모델 사용 예시
```python
from transformers import pipeline
import torch

model_id = "Seonghaa/korean-emotion-classifier-roberta"

device = 0 if torch.cuda.is_available() else -1  # GPU 있으면 0, 없으면 CPU(-1)

clf = pipeline(
    "text-classification",
    model=model_id,
    tokenizer=model_id,
    device=device
)

texts = [
    "오늘 길에서 10만원을 주웠어",
    "오늘 친구들이랑 노래방에 갔어",
    "오늘 시험 망쳤어",
]

for t in texts:
    pred = clf(t, truncation=True, max_length=256)[0]
    print(f"입력: {t}")
    print(f"→ 예측 감정: {pred['label']}, 점수: {pred['score']:.4f}
")

```
## 출력 예시:
입력: 오늘 길에서 10만원을 주웠어</br>
→ 예측 감정: 기쁨, 점수: 0.9619

입력: 오늘 친구들이랑 노래방에 갔어</br>
→ 예측 감정: 기쁨, 점수: 0.9653

입력: 오늘 시험 망쳤어</br>
→ 예측 감정: 슬픔, 점수: 0.9602

## 🛠️ 주요 특징

- 한국어 감정 분류 특화 라벨(6-class)

- 대규모 의사라벨링 기반 데이터 증강

- F1-score 중심 최적화와 과적합 방지 전략

- 클래스 불균형 보정을 통한 균형 잡힌 성능 확보

- Hugging Face Hub 업로드로 즉시 활용 가능
