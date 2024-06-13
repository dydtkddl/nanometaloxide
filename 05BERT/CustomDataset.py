import torch
from torch.utils.data import Dataset

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
