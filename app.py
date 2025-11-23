import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
import numpy as np

# 1. 保存したモデルとトークナイザを読み込む
MODEL_DIR = "./data"
print("モデルをロード中...")
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

# GPUがなければCPUで動かす
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # 評価モードへ

app = FastAPI()

# リクエストのデータ形式定義
class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "BERT API is running"}

@app.post("/predict")
def predict(req: TextRequest):
    text = req.text
    if not text:
        raise HTTPException(status_code=400, detail="テキストが空です")

    # 2. 推論処理（あなたの学習コードと同じ前処理）
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=52,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 確率に変換 (Softmax)
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # 最も高い確率のラベルIDを取得 (0 or 1)
        pred_label = torch.argmax(probs, dim=1).item()
        score = probs[0][pred_label].item()

    # 結果を返す
    # ※学習時のラベル定義に合わせて修正してください（例: 1=指示, 0=その他）
    is_instruction = (pred_label == 1) 
    
    return {
        "text": text,
        "is_instruction": is_instruction,
        "score": score
    }