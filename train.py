import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from model import load_model_lazy, unload_model
from transformers import get_linear_schedule_with_warmup

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        """
        Dataset for input texts.
        """
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
        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        return input_ids, attention_mask


def train_model_with_text(selected_model, custom_text, epochs=3, batch_size=8, learning_rate=5e-5):
    """
    Train a model using custom text.

    :param selected_model: Name of the selected model (e.g., "GPT2")
    :param custom_text: Custom text for training
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    :param learning_rate: Learning rate
    """
    # Load model and tokenizer
    model_data = load_model_lazy(selected_model)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    # Create dataset and dataloader
    dataset = TextDataset([custom_text], tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Start training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            # Move data to device (GPU or CPU)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Compute outputs and loss
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            # Backpropagation and weight update
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {step}/{len(dataloader)}, Loss: {loss.item()}")

        # Average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")

    # Save the trained model
    save_path = f"./trained_models/{selected_model}_custom_text"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model trained and saved to {save_path}")

    # Unload the model from memory
    unload_model(selected_model)