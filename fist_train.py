# -*- coding: utf-8 -*-
"""
KcELECTRA 학습 — SIMPLE 버전
- 검증에서 성능이 좋았던 구성으로 회귀
- 제거: LLRD / R-Drop / EMA  (정확도 하락 원인 제거)
- 유지: Label Smoothing(0.05), Grad Clip(1.0), MAX_LEN=224, EarlyStopping(patience=2)
- 추가: 클래스 가중치(class weights)로 불균형 보정
- 결과: macro-F1 기준 best 모델 저장

입력: golden_sample_filtered.json
출력: outputs_step1_kcelectra_SIMPLE/best_model/

단일 hold-out split (80/10/10) 방식
"""
import os, json, unicodedata as ud, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# ===== 설정 =====
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "beomi/KcELECTRA-base-v2022"
LABELS = ['분노','불안','슬픔','평온','당황','기쁨']
label2id = {k:i for i,k in enumerate(LABELS)}
id2label = {v:k for k,v in label2id.items()}

MAX_LEN = 224
BATCH = 16
LR = 2.5e-5          
WEIGHT_DECAY = 0.01
EPOCHS = 7
WARMUP_RATIO = 0.05
DROPOUT = 0.2
PATIENCE = 2
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0

DATA_DIR = "/content/drive/MyDrive/감정분류/data"
INPUT_JSON = os.path.join(DATA_DIR, "golden_sample_filtered.json")
OUT_DIR = os.path.join(DATA_DIR, "outputs_step1_kcelectra_SIMPLE")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 데이터 로딩 =====
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    rows = json.load(f)
raw_df = pd.DataFrame(rows)
assert {'user_input','emotion'}.issubset(raw_df.columns), "JSON에 user_input / emotion 필드가 필요합니다."

DEDUP_TEXT = True

def nfkc_keep(s: str) -> str:
    return ud.normalize('NFKC', str(s))

df = raw_df.rename(columns={'user_input':'text','emotion':'label'})[['text','label']].copy()
df['text'] = df['text'].astype(str).map(nfkc_keep)
df = df[df['label'].isin(LABELS)].reset_index(drop=True)
if DEDUP_TEXT:
    before = len(df); df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"[Dedup] removed {before - len(df)} duplicates (by text)")

df['label_id'] = df['label'].map(label2id)

# stratified 80/10/10
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label_id'])
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['label_id'])

print(f"[Split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
print("[Label Dist - train]\n", train_df['label'].value_counts())

# ===== 클래스 가중치 =====
cls_weights_np = compute_class_weight('balanced', classes=np.arange(len(LABELS)), y=train_df['label_id'].values)
cls_weights = torch.tensor(cls_weights_np, dtype=torch.float)
print("[Class Weights]", cls_weights_np)

# ===== 토크나이저 / 데이터셋 =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EmoDataset(Dataset):
    def __init__(self, frame, max_len):
        self.texts = frame['text'].tolist()
        self.labels = frame['label_id'].tolist()
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = tokenizer(self.texts[i], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['labels'] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

# DataLoader

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed); random.seed(worker_seed)

ds_train = EmoDataset(train_df, MAX_LEN)
ds_val   = EmoDataset(val_df,   MAX_LEN)
ds_test  = EmoDataset(test_df,  MAX_LEN)

gen = torch.Generator(); gen.manual_seed(SEED)

dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=gen)
dl_val   = DataLoader(ds_val,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker)
dl_test  = DataLoader(ds_test,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker)

# ===== 모델 / 옵티마이저 / 스케줄러 =====
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

t_total = EPOCHS * math.ceil(len(ds_train) / BATCH)
warmup_steps = max(1, int(t_total * WARMUP_RATIO))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

# ===== Accelerate 준비 =====
accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")
model, optimizer, dl_train, dl_val, dl_test, scheduler = accelerator.prepare(
    model, optimizer, dl_train, dl_val, dl_test, scheduler
)
cls_weights = cls_weights.to(accelerator.device)

# ===== 유틸: 평가 =====
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        inputs = {k: v for k,v in batch.items() if k in ['input_ids','attention_mask','token_type_ids']}
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(accelerator.gather(preds).cpu().numpy())
        all_labels.append(accelerator.gather(batch['labels']).cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=list(range(len(LABELS)))),
        "y_true": y_true, "y_pred": y_pred
    }

# ===== 학습 루프 (Label Smoothing + Class Weights + Early Stopping) =====
best_f1 = -1.0
no_improve = 0
for epoch in range(1, EPOCHS+1):
    model.train()
    for batch in dl_train:
        optimizer.zero_grad(set_to_none=True)
        inputs = {k: v for k,v in batch.items() if k in ['input_ids','attention_mask','token_type_ids']}
        logits = model(**inputs).logits
        n_cls = logits.size(-1)
        with torch.no_grad():
            onehot = F.one_hot(batch['labels'], n_cls).float()
            smooth = onehot * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / n_cls
        logp = F.log_softmax(logits, dim=-1)
        losses = -(smooth * logp).sum(dim=-1)
        # class weights
        w = cls_weights[batch['labels'].to(logits.device)]
        loss = (losses * w).mean()

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step(); scheduler.step()

    metrics = evaluate(model, dl_val)
    accelerator.print(f"[Epoch {epoch}] val_macro_f1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f}")
    if metrics['macro_f1'] > best_f1:
        best_f1 = metrics['macro_f1']; no_improve = 0
        unwrapped = accelerator.unwrap_model(model)
        save_dir = os.path.join(OUT_DIR, "best_model")
        accelerator.print(f"  ↳ Save best to {save_dir} (macro_f1={best_f1:.4f})")
        unwrapped.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            accelerator.print("Early stopping triggered.")
            break

# ===== 테스트 평가 =====
model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(OUT_DIR, "best_model")
).to(DEVICE)
model = accelerator.prepare(model)

m = evaluate(model, dl_test)
accelerator.print("\n테스트 성능")
accelerator.print(f"accuracy: {m['accuracy']:.6f}")
accelerator.print(f"macro_f1: {m['macro_f1']:.6f}")
accelerator.print(f"weighted_f1: {m['weighted_f1']:.6f}")

cm = confusion_matrix(m['y_true'], m['y_pred'], labels=list(range(len(LABELS))))
accelerator.print("\n[Confusion Matrix]\n", cm)
accelerator.print("\n[Per-class report]\n", classification_report(m['y_true'], m['y_pred'], target_names=LABELS, digits=4))

accelerator.print("\nStep 1 완료: SIMPLE 구성으로 학습/평가 종료")
