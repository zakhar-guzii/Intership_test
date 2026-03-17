import argparse
import logging
import sys
from pathlib import Path

from transformers import pipeline, Pipeline


logger = logging.getLogger(__name__)


def load_ner_pipeline(model_path: str, device: int = -1) -> Pipeline:
    """
    Load the fine-tuned NER pipeline from a local directory.

    Args:
        model_path: Path to the saved model directory (must contain
                    config.json, tokenizer files, and model weights).
        device:     -1 for CPU, 0+ for CUDA device index.

    Returns:
        A HuggingFace token-classification pipeline.
    """
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        logger.error(
            f"Model directory not found: '{model_path}'. "
            "Ensure --model_path points to the 'best_model' folder "
            "produced by ner/train.py."
        )
        sys.exit(1)

    logger.info(f"Loading NER model from: {model_path}")
    return pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple",
        device=device,
    )



def extract_animals(
    text: str,
    ner_pipeline: Pipeline,
    min_confidence: float = 0.0,
) -> list[dict]:
    """
    Run the NER pipeline on `text` and return all ANIMAL entities.

    Args:
        text:           Input string to analyse.
        ner_pipeline:   Pre-loaded pipeline instance (reuse across calls).
        min_confidence: Discard entities with score below this threshold.

    Returns:
        List of dicts with keys: word (str), score (float).
        Sorted by descending confidence score.
    """
    if not text or not text.strip():
        logger.warning("Received empty input text.")
        return []

    predictions = ner_pipeline(text)

    animals = [
        {"word": entity["word"].strip(), "score": round(entity["score"], 4)}
        for entity in predictions
        if "ANIMAL" in entity["entity_group"]
        and entity["score"] >= min_confidence
    ]

    seen: set[str] = set()
    unique_animals: list[dict] = []
    for a in animals:
        key = a["word"].lower()
        if key not in seen:
            seen.add(key)
            unique_animals.append(a)

    logger.debug(f"Raw predictions: {predictions}")
    logger.debug(f"Filtered animals: {unique_animals}")

    return sorted(unique_animals, key=lambda x: x["score"], reverse=True)


def extract_animal_names(
    text: str,
    ner_pipeline: Pipeline,
    min_confidence: float = 0.0,
) -> list[str]:
    """
    Convenience wrapper — returns only the animal name strings.
    Used by pipeline.py for downstream comparison with the CV model.
    """
    return [a["word"] for a in extract_animals(text, ner_pipeline, min_confidence)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract animal entity names from text using the fine-tuned NER model."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the fine-tuned NER model directory.",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Input text to analyse.",
    )
    parser.add_argument(
        "--text_file", type=str, default=None,
        help="Path to a plain-text file; each line is analysed separately.",
    )
    parser.add_argument(
        "--min_confidence", type=float, default=0.0,
        help="Minimum entity confidence score to include in results (0.0-1.0).",
    )
    parser.add_argument(
        "--device", type=int, default=-1,
        help="Device index: -1 for CPU, 0+ for CUDA GPU (default: -1).",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    if args.text is None and args.text_file is None:
        parser.error("Provide at least one of --text or --text_file.")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    ner_pipeline = load_ner_pipeline(args.model_path, device=args.device)

    lines: list[str] = []
    if args.text:
        lines.append(args.text)
    if args.text_file:
        file_path = Path(args.text_file)
        if not file_path.exists():
            logger.error(f"Text file not found: '{args.text_file}'")
            sys.exit(1)
        lines.extend(
            line.strip()
            for line in file_path.read_text().splitlines()
            if line.strip()
        )

    separator = "=" * 50
    print(separator)
    for line in lines:
        results = extract_animals(line, ner_pipeline, args.min_confidence)
        print(f"  Input  : {line}")
        if results:
            for a in results:
                print(f"  Animal : {a['word']:<15}  confidence: {a['score']:.4f}")
        else:
            print("  Animal : none detected")
        print("-" * 50)


if __name__ == "__main__":
    main()