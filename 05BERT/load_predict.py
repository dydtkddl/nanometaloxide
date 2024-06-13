import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import re
import numpy as np
from train import BERTTextClassifier
import json
import os

# 텍스트 전처리 함수
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("–", "-")
    text = re.sub(r'[^a-zA-Z0-9\s\-\_\(\)\[\]\/@]', '', text)
    return text

# 로짓 값을 라벨로 변환하는 함수
def to_label_text(predict_logit):
    names = [
        'ZnO',
        'SiO2',
        'TiO2',
        'Al2O3',
        'CuO',
        'Other'
    ]
    predict_names = []
    for i, logit in enumerate(predict_logit):
        if logit == 1:
            name = names[i]
            predict_names.append(name)
    return predict_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화 및 로드
bert_model_name = "./repos/BERTS/matbert-base-uncased"
model = BERTTextClassifier(nanoparticle_classes=6, subject_classes=6, bert_model_name=bert_model_name)
model.to(device)

# 저장된 모델 로드
model_path = "./repos/MODEL_WEIGHT/BERT_0607_each2/[BERT]BERTNAMEmatbert-base-uncased_processID24_06_07_15_56_21_4_Random_State13_lr2e-05_epoch15_threshold0.5_MAX_LEN128.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

# JSON 파일에서 제목 로드
import os
files = os.listdir("./repos/OXIDE/")
da = []
for i in files:
    if i[-4:] == "json":
        with open("./repos/OXIDE/%s"%i, "r") as f:
            da_ = json.load(f)
        da+=da_
# with open("./repos/OXIDE/titles_from84201.json", "r") as f:
#     da2 = json.load(f)
# with open("./repos/OXIDE/titles_from129701.json", "r") as f:
#     da3 = json.load(f)
# with open("./repos/OXIDE/titles_from259401.json", "r") as f:
#     da4 = json.load(f)
# with open("./repos/OXIDE/titles_from297801.json", "r") as f:
#     da4 = json.load(f)
# with open("./repos/OXIDE/titles_from397601.json", "r") as f:
#     da4 = json.load(f)
# da = da+ da2 + da3 + da4
print(len(da))
texts = [d["title"].replace("\n", "") for d in da]
dois = [d["doi"] for d in da]
publication_dates = [d["publicationDate"] for d in da]

# 예측 수행을 위한 텍스트 목록
# 전처리 및 토큰화
cleaned_texts = [clean_text(text) for text in texts]
inputs = tokenizer(cleaned_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

batch_size = 32  # 배치 사이즈 설정
num_batches = len(cleaned_texts) // batch_size + int(len(cleaned_texts) % batch_size != 0)
from tqdm import tqdm
# 로그 파일 생성
log_file_path = "./repos/OXIDE/prediction_log_%s.txt"%(len(da))
with open(log_file_path, "w") as log_file:
    for i in tqdm(range(num_batches)):
        batch_input_ids = inputs['input_ids'][i*batch_size:(i+1)*batch_size].to(device)
        batch_attention_mask = inputs['attention_mask'][i*batch_size:(i+1)*batch_size].to(device)
        
        with torch.no_grad():
            is_nanoparticle_logits, main_subject_logits = model(batch_input_ids, batch_attention_mask)
        
        for j in range(len(batch_input_ids)):
            text_idx = i*batch_size + j
            is_nanoparticle_result = (is_nanoparticle_logits[j] >= 0.5).int()
            main_subject_result = (main_subject_logits[j] >= 0.5).int()
            nanoparticle_predicted = to_label_text(is_nanoparticle_result.tolist())
            main_subject_predicted = to_label_text(main_subject_result.tolist())
            
            log_file.write(f"Text: {texts[text_idx]}\n")
            log_file.write(f"DOI: {dois[text_idx]}\n")
            log_file.write(f"Publication Date: {publication_dates[text_idx]}\n")
            log_file.write(f"Is Nanoparticle: {nanoparticle_predicted}\n")
            log_file.write(f"Main Subject: {main_subject_predicted}\n\n")
            # print(f"Text: {texts[text_idx]}")
            # print("Is Nanoparticle:", nanoparticle_predicted)
            # print("Main Subject:", main_subject_predicted)
            # print()
