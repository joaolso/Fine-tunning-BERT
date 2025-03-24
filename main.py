from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Carrega um dataset de exemplo (sentimentos)
dataset = load_dataset("imdb", split="train[:2000]")  # Apenas 2000 exemplos pra rodar rápido
dataset = dataset.train_test_split(test_size=0.2)

# Carrega o tokenizer e o modelo BERT pré-treinado
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Função para tokenizar os textos
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

df = dataset["train"].to_pandas()
df[["text", "label"]].to_csv("train_dataset_visivel.csv", index=False)

# Tokeniza os dados
dataset = dataset.map(tokenize, batched=True)

dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Define os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./bert-finetuned",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    logging_dir="./logs",
)

# Cria o Trainer (classe que cuida do fine-tuning)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Inicia o treinamento (fine-tuning)
trainer.train()

# Teste de inferência com texto novo
text = "This movie was really good!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=1)

print("Sentimento:", "Positivo" if prediction.item() == 1 else "Negativo")
