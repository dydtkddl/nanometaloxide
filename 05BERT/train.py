import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from EarlyStopping import EarlyStopping
from torch.utils.data import Dataset
## SSH의 경우 기본 path는 "./repos/OXIDE/"
class BERTTextClassifier(nn.Module):
    def __init__(self, nanoparticle_classes, subject_classes, bert_model_name):
        super(BERTTextClassifier, self).__init__()
        self.bert_nanoparticle = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=nanoparticle_classes)
        self.bert_main_subject = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=subject_classes)

    def forward(self, input_ids, attention_mask):
        outputs_is_nanoparticle = self.bert_nanoparticle(input_ids=input_ids, attention_mask=attention_mask)
        outputs_main_subject = self.bert_main_subject(input_ids=input_ids, attention_mask=attention_mask)
        return outputs_is_nanoparticle.logits, outputs_main_subject.logits

class CustomDataset(Dataset):
    def __init__(self, texts, is_nanoparticle_labels, main_subject_labels, tokenizer, max_len):
        self.texts = texts
        self.is_nanoparticle_labels = is_nanoparticle_labels
        self.main_subject_labels = main_subject_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        is_nanoparticle_label = torch.tensor(self.is_nanoparticle_labels[idx], dtype=torch.float)
        main_subject_label = torch.tensor(self.main_subject_labels[idx], dtype=torch.float)

        encoding = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'is_nanoparticle_labels': is_nanoparticle_label,
            'main_subject_labels': main_subject_label
        }

def main(args):
    print("BERT")
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    BERT = args.bert
    DATAPATH = args.data_path
    THRESHOLD = args.threshold
    RANDOM_STATE = args.random_state
    SSHPATH = args.ssh_path
    BERTNAME= args.BERTNAME
    PROCESSID= args.processID
    
    TRAIN_VAL_LOG_PATH = SSHPATH + args.train_val_log_path
    TRAIN_LOSS_AVG_IMAGE_PATH = SSHPATH + args.train_loss_avg_image_path
    VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH = SSHPATH + args.validation_metric_image_nanoparticle_path
    VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH = SSHPATH + args.validation_metric_image_main_subject_path
    VALIDATION_METRIC_IMAGE_AVG_PATH = SSHPATH + args.validation_metric_image_avg_path
    TEST_METRIC_LOG_PATH = SSHPATH + args.test_metric_log_path
    TEST_METRIC_IMAGE_NANOPARTICLE_PATH = SSHPATH + args.test_metric_image_nanoparticle_path
    TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH = SSHPATH + args.test_metric_image_main_subject_path
    TEST_METRIC_IMAGE_AVG_PATH = SSHPATH + args.test_metric_image_avg_path
    MODEL_WEIGHT_PATH = SSHPATH + args.model_weight_path
    PATHLIST = [TRAIN_VAL_LOG_PATH,
                TRAIN_LOSS_AVG_IMAGE_PATH,
                VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH,
                VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH,
                VALIDATION_METRIC_IMAGE_AVG_PATH,
                TEST_METRIC_LOG_PATH,
                TEST_METRIC_IMAGE_NANOPARTICLE_PATH,
                TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH,
                TEST_METRIC_IMAGE_AVG_PATH,
                MODEL_WEIGHT_PATH]
    for path__ in PATHLIST:
        if not os.path.exists(path__):
            os.makedirs(path__)
    print(
        "MAX_LEN : ", MAX_LEN, "\n",
        "BATCH_SIZE : ", BATCH_SIZE, "\n",
        "EPOCHS : ", EPOCHS, "\n",
        "LR : ", LR, "\n",
        "DATAPATH : ", DATAPATH, "\n",
        "BERT : ", BERT,
        "THRESHOLD : ", THRESHOLD,
        "RANDOM_STATE : ", RANDOM_STATE,
        "TRAIN_VAL_LOG_PATH : ", TRAIN_VAL_LOG_PATH,
        "TRAIN_LOSS_AVG_IMAGE_PATH : ", TRAIN_LOSS_AVG_IMAGE_PATH,
        "VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH : ", VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH,
        "VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH : ", VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH,
        "VALIDATION_METRIC_IMAGE_AVG_PATH : ", VALIDATION_METRIC_IMAGE_AVG_PATH,
        "TEST_METRIC_LOG_PATH : ", TEST_METRIC_LOG_PATH,
        "TEST_METRIC_IMAGE_NANOPARTICLE_PATH : ", TEST_METRIC_IMAGE_NANOPARTICLE_PATH,
        "TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH : ", TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH,
        "TEST_METRIC_IMAGE_AVG_PATH : ", TEST_METRIC_IMAGE_AVG_PATH,
        "MODEL_WEIGHT_PATH : ", MODEL_WEIGHT_PATH,
        "SSHPATH : ", SSHPATH,
        "BERTNAME : ", BERTNAME,
        "PROCESSID : ", PROCESSID,
    )

    tokenizer = BertTokenizerFast.from_pretrained(BERT)

    df = pd.read_excel(DATAPATH)
    texts = df['title'].tolist()
    is_nanoparticle_labels = [ast.literal_eval(sub) for sub in df['is_nanoparticle'].tolist()]
    main_subject_labels = [ast.literal_eval(sub) for sub in df['main_subject'].tolist()]

    train_texts, temp_texts, train_is_nanoparticle_labels, temp_is_nanoparticle_labels, train_main_subject_labels, temp_main_subject_labels = train_test_split(
        texts, is_nanoparticle_labels, main_subject_labels, test_size=0.2, random_state=RANDOM_STATE)
    val_texts, test_texts, val_is_nanoparticle_labels, test_is_nanoparticle_labels, val_main_subject_labels, test_main_subject_labels = train_test_split(
        temp_texts, temp_is_nanoparticle_labels, temp_main_subject_labels, test_size=0.5, random_state=RANDOM_STATE)

    train_dataset = CustomDataset(train_texts, train_is_nanoparticle_labels, train_main_subject_labels, tokenizer, max_len=MAX_LEN)
    val_dataset = CustomDataset(val_texts, val_is_nanoparticle_labels, val_main_subject_labels, tokenizer, max_len=MAX_LEN)
    test_dataset = CustomDataset(test_texts, test_is_nanoparticle_labels, test_main_subject_labels, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_nanoparticle_classes = 6
    num_subject_classes = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTTextClassifier(num_nanoparticle_classes, num_subject_classes, BERT)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    train_losses = []
    val_losses = []
    val_precisions_nanoparticle = []
    val_recalls_nanoparticle = []
    val_f1s_nanoparticle = []
    val_accuracies_nanoparticle = []
    val_precisions_main_subject = []
    val_recalls_main_subject = []
    val_f1s_main_subject = []
    val_accuracies_main_subject = []
    val_avg_precisions = []
    val_avg_recalls = []
    val_avg_f1s = []
    val_avg_accuracies = []

    log_file = open(os.path.join(TRAIN_VAL_LOG_PATH, "training_log.txt"), "w")

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            outputs_is_nanoparticle, outputs_main_subject = model(inputs, masks)
            loss_is_nanoparticle = criterion(outputs_is_nanoparticle, batch['is_nanoparticle_labels'].to(device))
            loss_main_subject = criterion(outputs_main_subject, batch['main_subject_labels'].to(device))
            loss = loss_is_nanoparticle + loss_main_subject
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0
        all_val_is_nanoparticle_labels = []
        all_val_main_subject_labels = []
        all_val_is_nanoparticle_preds = []
        all_val_main_subject_preds = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                outputs_is_nanoparticle, outputs_main_subject = model(inputs, masks)
                loss_is_nanoparticle = criterion(outputs_is_nanoparticle, batch['is_nanoparticle_labels'].to(device))
                loss_main_subject = criterion(outputs_main_subject, batch['main_subject_labels'].to(device))
                loss = loss_is_nanoparticle + loss_main_subject
                total_val_loss += loss.item()

                all_val_is_nanoparticle_labels.append(batch['is_nanoparticle_labels'].cpu().numpy())
                all_val_main_subject_labels.append(batch['main_subject_labels'].cpu().numpy())
                all_val_is_nanoparticle_preds.append(torch.sigmoid(outputs_is_nanoparticle).cpu().numpy())
                all_val_main_subject_preds.append(torch.sigmoid(outputs_main_subject).cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss}")

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        all_val_is_nanoparticle_labels = np.concatenate(all_val_is_nanoparticle_labels)
        all_val_main_subject_labels = np.concatenate(all_val_main_subject_labels)
        all_val_is_nanoparticle_preds = (np.concatenate(all_val_is_nanoparticle_preds) >= THRESHOLD).astype(int)
        all_val_main_subject_preds = (np.concatenate(all_val_main_subject_preds) >= THRESHOLD).astype(int)

        precision_nanoparticle = precision_score(all_val_is_nanoparticle_labels, all_val_is_nanoparticle_preds, average='macro', zero_division=0)
        recall_nanoparticle = recall_score(all_val_is_nanoparticle_labels, all_val_is_nanoparticle_preds, average='macro', zero_division=0)
        f1_nanoparticle = f1_score(all_val_is_nanoparticle_labels, all_val_is_nanoparticle_preds, average='macro', zero_division=0)
        accuracy_nanoparticle = accuracy_score(all_val_is_nanoparticle_labels, all_val_is_nanoparticle_preds)

        precision_main_subject = precision_score(all_val_main_subject_labels, all_val_main_subject_preds, average='macro', zero_division=0)
        recall_main_subject = recall_score(all_val_main_subject_labels, all_val_main_subject_preds, average='macro', zero_division=0)
        f1_main_subject = f1_score(all_val_main_subject_labels, all_val_main_subject_preds, average='macro', zero_division=0)
        accuracy_main_subject = accuracy_score(all_val_main_subject_labels, all_val_main_subject_preds)

        val_precisions_nanoparticle.append(precision_nanoparticle)
        val_recalls_nanoparticle.append(recall_nanoparticle)
        val_f1s_nanoparticle.append(f1_nanoparticle)
        val_accuracies_nanoparticle.append(accuracy_nanoparticle)

        val_precisions_main_subject.append(precision_main_subject)
        val_recalls_main_subject.append(recall_main_subject)
        val_f1s_main_subject.append(f1_main_subject)
        val_accuracies_main_subject.append(accuracy_main_subject)

        precision = (precision_main_subject + precision_nanoparticle) / 2
        recall = (recall_main_subject + recall_nanoparticle) / 2
        f1 = (f1_main_subject + f1_nanoparticle) / 2
        accuracy = (accuracy_main_subject + accuracy_nanoparticle) / 2

        val_avg_precisions.append(precision)
        val_avg_recalls.append(recall)
        val_avg_f1s.append(f1)
        val_avg_accuracies.append(accuracy)

        log_file.write(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, "
                       f"Precision (Nanoparticle): {precision_nanoparticle}, Recall (Nanoparticle): {recall_nanoparticle}, "
                       f"F1 (Nanoparticle): {f1_nanoparticle}, Accuracy (Nanoparticle): {accuracy_nanoparticle}, "
                       f"Precision (Main Subject): {precision_main_subject}, Recall (Main Subject): {recall_main_subject}, "
                       f"F1 (Main Subject): {f1_main_subject}, Accuracy (Main Subject): {accuracy_main_subject}, "
                       f"AVGPrecision: {precision}, AVGRecall: {recall}, AVGF1: {f1}, AVGAccuracy: {accuracy}\n")

    log_file.close()
    SAVE_FILE_NAME = f"[BERT]BERTNAME{BERTNAME}_processID{PROCESSID}_Random_State{RANDOM_STATE}_lr{LR}_epoch{epoch}_threshold{THRESHOLD}_MAX_LEN{MAX_LEN}"

    # Create plot directory if not exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Plot train and validation loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylim(0, 1)
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(TRAIN_LOSS_AVG_IMAGE_PATH, SAVE_FILE_NAME + ".jpg"))
    plt.close()

    # Plot validation metrics for nanoparticle
    plt.figure()
    plt.plot(range(1, len(val_precisions_nanoparticle) + 1), val_precisions_nanoparticle, label='Precision (Nanoparticle)')
    plt.plot(range(1, len(val_recalls_nanoparticle) + 1), val_recalls_nanoparticle, label='Recall (Nanoparticle)')
    plt.plot(range(1, len(val_f1s_nanoparticle) + 1), val_f1s_nanoparticle, label='F1 Score (Nanoparticle)')
    plt.plot(range(1, len(val_accuracies_nanoparticle) + 1), val_accuracies_nanoparticle, label='Accuracy (Nanoparticle)')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.ylim(0, 1)
    plt.title('Validation Metrics Over Epochs (Nanoparticle)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH, SAVE_FILE_NAME + ".jpg"))
    plt.close()

    # Plot validation metrics for main subject
    plt.figure()
    plt.plot(range(1, len(val_precisions_main_subject) + 1), val_precisions_main_subject, label='Precision (Main Subject)')
    plt.plot(range(1, len(val_recalls_main_subject) + 1), val_recalls_main_subject, label='Recall (Main Subject)')
    plt.plot(range(1, len(val_f1s_main_subject) + 1), val_f1s_main_subject, label='F1 Score (Main Subject)')
    plt.plot(range(1, len(val_accuracies_main_subject) + 1), val_accuracies_main_subject, label='Accuracy (Main Subject)')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.ylim(0, 1)
    plt.title('Validation Metrics Over Epochs (Main Subject)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH, SAVE_FILE_NAME + ".jpg"))
    plt.close()

    # Plot validation metrics (average)
    plt.figure()
    plt.plot(range(1, len(val_avg_precisions) + 1), val_avg_precisions, label='Precision')
    plt.plot(range(1, len(val_avg_recalls) + 1), val_avg_recalls, label='Recall')
    plt.plot(range(1, len(val_avg_f1s) + 1), val_avg_f1s, label='F1 Score')
    plt.plot(range(1, len(val_avg_accuracies) + 1), val_avg_accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.ylim(0, 1)
    plt.title('Validation Metrics Over Epochs (AVG)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(VALIDATION_METRIC_IMAGE_AVG_PATH, SAVE_FILE_NAME + ".jpg"))
    plt.close()

    # Evaluate on test set
    model.eval()
    all_test_is_nanoparticle_labels = []
    all_test_main_subject_labels = []
    all_test_is_nanoparticle_preds = []
    all_test_main_subject_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            outputs_is_nanoparticle, outputs_main_subject = model(inputs, masks)
            all_test_is_nanoparticle_labels.append(batch['is_nanoparticle_labels'].cpu().numpy())
            all_test_main_subject_labels.append(batch['main_subject_labels'].cpu().numpy())
            all_test_is_nanoparticle_preds.append(torch.sigmoid(outputs_is_nanoparticle).cpu().numpy())
            all_test_main_subject_preds.append(torch.sigmoid(outputs_main_subject).cpu().numpy())

    all_test_is_nanoparticle_labels = np.concatenate(all_test_is_nanoparticle_labels)
    all_test_main_subject_labels = np.concatenate(all_test_main_subject_labels)
    all_test_is_nanoparticle_preds = (np.concatenate(all_test_is_nanoparticle_preds) >= THRESHOLD).astype(int)
    all_test_main_subject_preds = (np.concatenate(all_test_main_subject_preds) >= THRESHOLD).astype(int)

    # Calculate test metrics for nanoparticle
    precision_nanoparticle = precision_score(all_test_is_nanoparticle_labels, all_test_is_nanoparticle_preds, average='macro', zero_division=0)
    recall_nanoparticle = recall_score(all_test_is_nanoparticle_labels, all_test_is_nanoparticle_preds, average='macro', zero_division=0)
    f1_nanoparticle = f1_score(all_test_is_nanoparticle_labels, all_test_is_nanoparticle_preds, average='macro', zero_division=0)
    accuracy_nanoparticle = accuracy_score(all_test_is_nanoparticle_labels, all_test_is_nanoparticle_preds)

    # Calculate test metrics for main subject
    precision_main_subject = precision_score(all_test_main_subject_labels, all_test_main_subject_preds, average='macro', zero_division=0)
    recall_main_subject = recall_score(all_test_main_subject_labels, all_test_main_subject_preds, average='macro', zero_division=0)
    f1_main_subject = f1_score(all_test_main_subject_labels, all_test_main_subject_preds, average='macro', zero_division=0)
    accuracy_main_subject = accuracy_score(all_test_main_subject_labels, all_test_main_subject_preds)
    print(all_test_main_subject_labels)
    '''[[0. 0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]
 [1. 0. 1. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]]
'''
    # Combine test metrics
    both_label = []
    for nano_label, ma_label in zip(all_test_is_nanoparticle_labels, all_test_main_subject_labels):
        lll = []
        for n, m in zip(nano_label, ma_label):
            if n == 1 and m == 1:
                lll.append(1)
            else:
                lll.append(0)
        both_label.append(lll)
    both_pred = []
    for nano_pred, ma_pred in zip(all_test_is_nanoparticle_preds, all_test_main_subject_preds):
        lll = []
        for n, m in zip(nano_pred, ma_pred):
            if n == 1 and m == 1:
                lll.append(1)
            else:
                lll.append(0)
        both_pred.append(lll)
    precision = precision_score(both_label, both_pred, average='macro', zero_division=0)
    recall = recall_score(both_label, both_pred, average='macro', zero_division=0)
    f1 = f1_score(both_label, both_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(both_label, both_pred)

    # Log test metrics
    with open(os.path.join(TEST_METRIC_LOG_PATH, SAVE_FILE_NAME + ".txt"), "w") as f:
        f.write(f"Test Precision (Nanoparticle): {precision_nanoparticle}\n")
        f.write(f"Test Recall (Nanoparticle): {recall_nanoparticle}\n")
        f.write(f"Test F1 Score (Nanoparticle): {f1_nanoparticle}\n")
        f.write(f"Test Accuracy (Nanoparticle): {accuracy_nanoparticle}\n")
        f.write(f"Test Precision (Main Subject): {precision_main_subject}\n")
        f.write(f"Test Recall (Main Subject): {recall_main_subject}\n")
        f.write(f"Test F1 Score (Main Subject): {f1_main_subject}\n")
        f.write(f"Test Accuracy (Main Subject): {accuracy_main_subject}\n")
        f.write(f"Test Precision (AVG): {precision}\n")
        f.write(f"Test Recall (AVG): {recall}\n")
        f.write(f"Test F1 Score (AVG): {f1}\n")
        f.write(f"Test Accuracy (AVG): {accuracy}\n")

    print(f"Test Precision (Nanoparticle): {precision_nanoparticle}")
    print(f"Test Recall (Nanoparticle): {recall_nanoparticle}")
    print(f"Test F1 Score (Nanoparticle): {f1_nanoparticle}")
    print(f"Test Accuracy (Nanoparticle): {accuracy_nanoparticle}")
    print(f"Test Precision (Main Subject): {precision_main_subject}")
    print(f"Test Recall (Main Subject): {recall_main_subject}")
    print(f"Test F1 Score (Main Subject): {f1_main_subject}")
    print(f"Test Accuracy (Main Subject): {accuracy_main_subject}")
    print(f"Test Precision (AVG): {precision}")
    print(f"Test Recall (AVG): {recall}")
    print(f"Test F1 Score (AVG): {f1}")
    print(f"Test Accuracy (AVG): {accuracy}")

    # Plot test metrics for nanoparticle
    metrics_nanoparticle = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values_nanoparticle = [precision_nanoparticle, recall_nanoparticle, f1_nanoparticle, accuracy_nanoparticle]
    
    plt.figure()
    plt.bar(metrics_nanoparticle, values_nanoparticle)
    plt.ylim(0, 1)
    plt.title('Test Metrics (Nanoparticle)')
    plt.grid()
    plt.savefig(os.path.join(TEST_METRIC_IMAGE_NANOPARTICLE_PATH, SAVE_FILE_NAME + ".jpg"))
    plt.close()

    # Plot test metrics for main subject
    metrics_main_subject = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values_main_subject = [precision_main_subject, recall_main_subject, f1_main_subject, accuracy_main_subject]

    plt.figure()
    plt.bar(metrics_main_subject, values_main_subject)
    plt.ylim(0, 1)
    plt.title('Test Metrics (Main Subject)')
    plt.grid()
    plt.savefig(os.path.join(TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH, SAVE_FILE_NAME + ".jpg"))
    plt.close()

    # Plot test metrics (average)
    metrics_avg = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values_avg = [precision, recall, f1, accuracy]

    plt.figure()
    plt.bar(metrics_avg, values_avg)
    plt.ylim(0, 1)
    plt.title('Test Metrics')
    plt.grid()
    plt.savefig(os.path.join(TEST_METRIC_IMAGE_AVG_PATH, SAVE_FILE_NAME + ".jpg"))
    plt.close()

    # Save model
    torch.save(model.state_dict(), os.path.join(MODEL_WEIGHT_PATH, SAVE_FILE_NAME + ".pth"))
    print("Model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT-based text classifier model.")
    parser.add_argument("--data_path", type=str, default="../04datapurify/purify_data.xlsx", help="Path to the dataset file")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length of the input sequences")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--bert", type=str, default="bert-base-uncased", help="BERT model name")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for data splitting")

    parser.add_argument("--train_val_log_path", type=str, default="./", help="Path to save train/validation logs")
    parser.add_argument("--train_loss_avg_image_path", type=str, default="./", help="Path to save train loss and average image")
    parser.add_argument("--validation_metric_image_nanoparticle_path", type=str, default="./", help="Path to save validation metrics image for nanoparticle")
    parser.add_argument("--validation_metric_image_main_subject_path", type=str, default="./", help="Path to save validation metrics image for main subject")
    parser.add_argument("--validation_metric_image_avg_path", type=str, default="./", help="Path to save validation metrics image (average)")
    parser.add_argument("--test_metric_log_path", type=str, default="./", help="Path to save test metrics log")
    parser.add_argument("--test_metric_image_nanoparticle_path", type=str, default="./", help="Path to save test metrics image for nanoparticle")
    parser.add_argument("--test_metric_image_main_subject_path", type=str, default="./", help="Path to save test metrics image for main subject")
    parser.add_argument("--test_metric_image_avg_path", type=str, default="./", help="Path to save test metrics image (average)")
    parser.add_argument("--model_weight_path", type=str, default="./", help="Path to save model weights")
    
    parser.add_argument("--ssh_path", type=str, default=None, help="Path SSH remote folder")
    parser.add_argument("--BERTNAME", type=str, default=None, help="Path SSH BERTNAME folder")
    parser.add_argument("--processID", type=str, default=None, help="Path SSH PROCESSID folder")

    args = parser.parse_args()
    main(args)


