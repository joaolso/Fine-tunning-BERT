# Fine-tuning e Inferência com TinyBERT

Este projeto realiza o fine-tuning do modelo **TinyBERT** com o dataset de sentimentos **IMDB**, usando a biblioteca **Hugging Face Transformers**.

---

## 📁 Estrutura do Projeto

```bash
fine-tuning-tinybert/
├── main_finetune.py         # Script principal com treino e inferência
├── models/                  # Onde o modelo fine-tuned é salvo
├── logs/                    # Logs de treino do Hugging Face
├── datasets/                # (Opcional) CSVs exportados para inspeção
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```

---

## 🚀 Como usar o script principal

### 1. Instale as dependências

Com `uv` (recomendado):
```bash
uv pip install -r requirements.txt
```

Ou com `pip`:
```bash
pip install -r requirements.txt
```

---

### 2. Treinar o modelo

```bash
python main_finetune.py --train
```

Isso vai:
- Carregar TinyBERT
- Tokenizar o dataset IMDB (2000 amostras)
- Fazer fine-tuning por 2 épocas
- Salvar o modelo em `tinybert-finetuned-imdb/`

---

### 3. Fazer inferência com texto novo

```bash
python main_finetune.py --infer --text "This movie was really good!"
```

Saída:
```bash
🔍 Realizando inferência...
📢 Sentimento detectado: Positivo
```

---

### 4. Treinar e inferir de uma vez:

```bash
python main_finetune.py --train --infer --text "A fantastic and inspiring story."
```

---

## 📂 Modelos e Tokenizers

- Modelo base: `huawei-noah/TinyBERT_General_4L_312D`
- Tokenizer e modelo são salvos em: `tinybert-finetuned-imdb/`

---

## ⚙️ Argumentos disponíveis no script

| Flag             | Descrição                                         |
|------------------|--------------------------------------------------|
| `--train`        | Executa o fine-tuning                             |
| `--infer`        | Executa inferência com o modelo salvo            |
| `--text`         | Frase para classificação de sentimento          |
| `--model_name`   | Nome do modelo base pré-treinado (opcional)     |
| `--model_path`   | Caminho onde o modelo fine-tuned está salvo      |

---

## 📚 Requisitos (requirements.txt)

```txt
transformers
datasets
torch
scikit-learn
```

---

Se quiser, podemos adicionar suporte a wandb, exportação de métricas, ou uma interface com Streamlit ✨