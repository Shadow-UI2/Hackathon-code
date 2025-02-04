import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Your custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the input text
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        
        # Remove the batch dimension for input tensors
        input_ids = encoding['input_ids'].squeeze(0)  # Remove the batch dimension (1,)
        attention_mask = encoding['attention_mask'].squeeze(0)  # Remove the batch dimension (1,)
        
        # Prepare the input dictionary
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(self.labels[idx])
        }
        return item

# Function to retrain the model with the updated dataset
def retrain_model(model, tokenizer, texts, labels):
    # Create dataset with updated texts and labels
    train_dataset = TextDataset(texts, labels, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        evaluation_strategy="epoch",     # evaluation strategy
        learning_rate=2e-5,              # learning rate
        per_device_train_batch_size=8,   # batch size for training
        per_device_eval_batch_size=8,    # batch size for evaluation
        num_train_epochs=3,              # number of training epochs
        weight_decay=0.01,               # strength of weight decay
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,                         # the model to be trained
        args=training_args,                  # training arguments
        train_dataset=train_dataset,         # training dataset
        eval_dataset=train_dataset,          # evaluation dataset (you can separate eval set if needed)
    )
    
    # Train the model
    trainer.train()

# Function to classify user input and retrain the model if needed
def classify_and_retrain():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # Example texts and labels (replace with your dataset)
    texts = ["Some sample text", "Another example of AI generated content"]
    labels = [0, 1]  # 0 = Human, 1 = AI-Generated
    
    while True:
        # Get user input
        text = input("Enter text to classify (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            break

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the predicted label (0 = Human, 1 = AI-Generated)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs).item()
        
        # Print prediction
        print(f"Model Prediction: {'Human' if pred_label == 0 else 'AI-Generated'} (Confidence: {probs[0][pred_label]:.2f})")

        # Get user feedback on the prediction
        feedback = input("Was this prediction correct? (yes/no): ").lower()
        
        if feedback == 'no':
            # Collect the correct label and retrain
            correct_label = input("Please provide the correct label (Human/AI-Generated): ").lower()
            label = 0 if correct_label == 'human' else 1

            # Store the new example and label (append to the dataset)
            texts.append(text)
            labels.append(label)

            # Retrain the model with the updated dataset
            retrain_model(model, tokenizer, texts, labels)

# Start the classification and retraining loop
if __name__ == "__main__":
    classify_and_retrain()
