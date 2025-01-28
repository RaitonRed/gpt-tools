import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from model import load_model_lazy, unload_model
from database import fetch_all_inputs, clear_database

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        attention_mask = encodings.attention_mask.squeeze(0)
        return encodings.input_ids.squeeze(0), attention_mask

def train_model_with_text(selected_model, custom_text, epochs, batch_size):
    """
    آموزش مدل با متن سفارشی.
    """
    model_data = load_model_lazy(selected_model)  # بارگذاری مدل
    model = model_data["model"]  # استخراج مدل
    tokenizer = model_data["tokenizer"]  # استخراج توکنایزر

    dataset = TextDataset([custom_text], tokenizer)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    _train_model(model, tokenizer, dataloader, epochs, selected_model, "custom_text")
    unload_model(selected_model)  # تخلیه مدل پس از استفاده

def _train_model(model, tokenizer, dataloader, epochs, model_name, method):
    """
    منطق مشترک آموزش مدل.
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # انتقال مدل به GPU در صورت وجود
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # محاسبه خروجی و خطا
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    # ذخیره مدل
    save_path = f"trained_{model_name}_{method}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model {model_name} trained with {method} and saved to {save_path}.")