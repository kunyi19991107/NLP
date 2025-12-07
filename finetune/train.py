import argparse
import logging
import os
import json

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from constant import PRIMARY_CATEGORY_TO_INDEX, INDEX_TO_PRIMARY_CATEGORY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    preds = logits.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on JSON data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to json file")
    parser.add_argument("--model_name", type=str, default="ProsusAI/finbert")
    parser.add_argument("--output_dir", type=str, default="./model_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # -----------------------------
    # Load JSON
    # -----------------------------
    logger.info(f"Loading data from {args.data_path}...")
    with open(args.data_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame({
        "text": [d["risk_factors"] for d in data],
        "category": [d["primary_risk_factor"][0] for d in data] # only use the first primary risk factor
    })

    # -----------------------------
    # Encode labels
    # -----------------------------
    category_to_id = PRIMARY_CATEGORY_TO_INDEX
    id_to_category = INDEX_TO_PRIMARY_CATEGORY
    df["labels"] = df["category"].map(category_to_id)
    num_labels = len(category_to_id)

    logger.info(f"Detected {num_labels} categories.")
    logger.info(f"Categories: {category_to_id}")

    # -----------------------------
    # Train/val split
    # -----------------------------
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42 #, stratify=df["labels"]
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=256
        )

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_val = val_dataset.map(tokenize, batched=True)

    keep_cols = ["input_ids", "attention_mask", "labels"]
    tokenized_train = tokenized_train.remove_columns(
        [c for c in tokenized_train.column_names if c not in keep_cols]
    )
    tokenized_val = tokenized_val.remove_columns(
        [c for c in tokenized_val.column_names if c not in keep_cols]
    )

    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    # -----------------------------
    # Model with resized classifier head
    # -----------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        id2label=id_to_category,
        label2id=category_to_id   
    )

    # -----------------------------
    # Training args
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        use_mps_device=torch.backends.mps.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    # -----------------------------
    # Train
    # -----------------------------
    logger.info("Starting training...")
    trainer.train()

    # -----------------------------
    # Save
    # -----------------------------
    logger.info(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump(
            {"category_to_id": category_to_id, "id_to_category": id_to_category},
            f,
            indent=2
        )


if __name__ == "__main__":
    main()
