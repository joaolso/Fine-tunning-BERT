import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import argparse
import os

def load_and_tokenize_dataset(model_name, sample_size=2000):
    print("🔄 Carregando e tokenizando dataset...")
    dataset = load_dataset("imdb", split=f"train[:{sample_size}]").train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Salvar dataset original visível (text + label)
    os.makedirs("datasets", exist_ok=True)
    print("📝 Salvando dataset visível para auditoria...")
    df_visivel = dataset["train"].to_pandas()[["text", "label"]]
    df_visivel.to_csv("datasets/train_dataset_visivel.csv", index=False)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return dataset, tokenizer

def train_model(model_name, output_dir="tinybert-finetuned-imdb", epochs=2):
    dataset, tokenizer = load_and_tokenize_dataset(model_name)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    print("🚀 Iniciando o fine-tuning...")
    trainer.train()

    print("💾 Salvando modelo e tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Fine-tuning concluído!")

def run_inference(model_path, text):
    print("🔍 Realizando inferência...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)

    sentimento = "Positivo" if prediction.item() == 1 else "Negativo"
    print(f"📢 Sentimento detectado: {sentimento}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning + Inferência com TinyBERT")
    parser.add_argument("--train", action="store_true", help="Executar o fine-tuning")
    parser.add_argument("--infer", action="store_true", help="Executar inferência")
    parser.add_argument("--text", type=str, default="This movie was really good!", help="Texto para inferência")
    parser.add_argument("--model_name", type=str, default="huawei-noah/TinyBERT_General_4L_312D", help="Nome do modelo base")
    parser.add_argument("--model_path", type=str, default="tinybert-finetuned-imdb", help="Caminho do modelo salvo")

    args = parser.parse_args()

    if args.train:
        train_model(args.model_name, output_dir=args.model_path)

    if args.infer:
        run_inference(args.model_path, args.text)
