import torch
from transformers import BertTokenizerFast
import re
import numpy as np
import json
import os
from train import BERTTextClassifier  # Ensure this import matches your actual module/file name
from tqdm import tqdm

# Function to clean the text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("–", "-")
    text = re.sub(r'[^a-zA-Z0-9\s\-\_\(\)\[\]\/@]', '', text)
    return text

# Function to convert logits to label text
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

# Initialize and load the model
bert_model_name = "./repos/BERTS/matbert-base-uncased"
model = BERTTextClassifier(nanoparticle_classes=6, subject_classes=6, bert_model_name=bert_model_name)
model.to(device)

# Load the saved model weights
model_path = "./repos/MODEL_WEIGHT/BERT_0607_each2/[BERT]BERTNAMEmatbert-base-uncased_processID24_06_07_15_56_21_4_Random_State13_lr2e-05_epoch15_threshold0.5_MAX_LEN128.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

# Load titles from JSON files
files = os.listdir("./repos/OXIDE/")
data = []
for file in files:
    if file.endswith(".json"):
        with open(os.path.join("./repos/OXIDE/", file), "r") as f:
            data += json.load(f)

print(f"Total records: {len(data)}")
texts = [d["title"].replace("\n", "") for d in data]
dois = [d["doi"] for d in data]
publication_dates = [d["publicationDate"] for d in data]

# Clean and tokenize the texts
cleaned_texts = [clean_text(text) for text in texts]
inputs = tokenizer(cleaned_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

batch_size = 32  # Batch size
num_batches = len(cleaned_texts) // batch_size + int(len(cleaned_texts) % batch_size != 0)

# Create log file
log_file_path = "./repos/OXIDE/prediction_log.txt"
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
