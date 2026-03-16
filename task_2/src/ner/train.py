import argparse
import logging
import os

import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate


LABEL_LIST = ["O", "B-ANIMAL", "I-ANIMAL"]
ID2LABEL   = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID   = {label: i for i, label in enumerate(LABEL_LIST)}



def tokenize_and_align_labels(
    examples: dict,
    tokenizer: AutoTokenizer,
    label_all_subwords: bool = False,
) -> dict:
    """
    Tokenise pre-split tokens and align NER tags with subword tokens.

    Args:
        examples:          Batch from a HuggingFace Dataset (keys: tokens, ner_tags).
        tokenizer:         Tokenizer matching the pretrained model.
        label_all_subwords: If True, propagate the label to every subword token.
                            If False (default / IOB2 convention), only the first
                            subword receives the label; continuations get -100
                            so they are ignored by CrossEntropyLoss.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
    )

    all_labels = []
    for i, label_ids_orig in enumerate(examples["ner_tags"]):
        word_ids          = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids         = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_ids_orig[word_idx])
            else:
                label_ids.append(
                    label_ids_orig[word_idx] if label_all_subwords else -100
                )
            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def build_compute_metrics(label_list: list[str]):
    """
    Returns a compute_metrics function for seqeval token-level NER evaluation.
    Reports precision, recall, and F1 per entity type plus macro averages.
    """
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_preds) -> dict:
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_labels = [
            [label_list[l] for l in label_row if l != -100]
            for label_row in labels
        ]
        true_preds = [
            [label_list[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": round(results["overall_precision"], 4),
            "recall":    round(results["overall_recall"],    4),
            "f1":        round(results["overall_f1"],        4),
            "accuracy":  round(results["overall_accuracy"],  4),
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer NER model for animal entity extraction."
    )
    parser.add_argument("--model_name",    type=str,   default="dslim/bert-base-NER",
                        help="Pretrained HuggingFace model name or local path.")
    parser.add_argument("--data_path",     type=str,   required=True,
                        help="Path to the training dataset (JSON lines format).")
    parser.add_argument("--val_path",      type=str,   default=None,
                        help="Path to a separate validation dataset (optional). "
                             "If omitted, 10%% of --data_path is used.")
    parser.add_argument("--output_path",   type=str,   default="models/ner_animal",
                        help="Directory to save checkpoints and the final model.")
    parser.add_argument("--epochs",        type=int,   default=5,
                        help="Maximum number of training epochs.")
    parser.add_argument("--batch_size",    type=int,   default=16,
                        help="Per-device batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Peak learning rate for AdamW.")
    parser.add_argument("--weight_decay",  type=float, default=0.01,
                        help="L2 weight decay coefficient.")
    parser.add_argument("--warmup_ratio",  type=float, default=0.1,
                        help="Fraction of total steps used for LR warm-up.")
    parser.add_argument("--patience",      type=int,   default=3,
                        help="Early-stopping patience (epochs without F1 improvement).")
    parser.add_argument("--val_split",     type=float, default=0.1,
                        help="Fraction of training data to use as validation "
                             "when --val_path is not provided.")
    parser.add_argument("--label_all_subwords", action="store_true",
                        help="Propagate NER label to all subword tokens "
                             "(default: first subword only, IOB2 convention).")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)

    if args.val_path:
        raw_dataset: DatasetDict = load_dataset(
            "json",
            data_files={"train": args.data_path, "validation": args.val_path},
        )
        logging.info(
            f"Loaded {len(raw_dataset['train'])} train / "
            f"{len(raw_dataset['validation'])} val examples."
        )
    else:
        full = load_dataset("json", data_files={"train": args.data_path})["train"]
        split = full.train_test_split(test_size=args.val_split, seed=args.seed)
        raw_dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
        logging.info(
            f"Auto-split: {len(raw_dataset['train'])} train / "
            f"{len(raw_dataset['validation'])} val examples "
            f"({int(args.val_split * 100)}% val)."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, args.label_all_subwords),
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir                  = args.output_path,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate               = args.learning_rate,
        weight_decay                = args.weight_decay,
        warmup_ratio                = args.warmup_ratio,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 20,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        seed                        = args.seed,
        report_to                   = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized_dataset["train"],
        eval_dataset    = tokenized_dataset["validation"],
        tokenizer       = tokenizer,
        data_collator   = DataCollatorForTokenClassification(tokenizer),
        compute_metrics = build_compute_metrics(LABEL_LIST),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )


    logging.info("\nStarting training...")
    trainer.train()

    best_model_path = os.path.join(args.output_path, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    logging.info(f"\nBest model saved to: {best_model_path}")

    metrics = trainer.evaluate()
    logging.info("\nFinal validation metrics:")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()