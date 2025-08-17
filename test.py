from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 저장된 모델 경로 (학습 끝나고 저장했던 best_model 폴더)
model_path = "data/outputs_trainer_final2/best_model"

# 토크나이저 & 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 감정 분류 파이프라인 생성
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 라벨 순서 (학습할 때 쓴 라벨)
LABELS = ['분노','불안','슬픔','평온','당황','기쁨']

# ===== 테스트 =====
texts = [
    "오늘 길에서 10만원을 주웠어",
    "오늘 친구들이랑 노래방에 갔어",
    "오늘 시험 망쳤어",
]

for t in texts:
    pred = emotion_classifier(t, truncation=True)[0]
    label = pred["label"] if "label" in pred else LABELS[pred["id"]]  # 혹시 label key가 없을 때 대비
    score = pred["score"]
    print(f"입력: {t}")
    print(f"→ 예측 감정: {label}, 점수: {score:.4f}\n")
