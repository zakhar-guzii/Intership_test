import argparse
import logging
import sys
from pathlib import Path

import torch

from src.classifier.inference import load_model as load_cv_model
from src.classifier.inference import predict as predict_cv
from src.ner.inference import load_ner_pipeline, extract_animal_names


logger = logging.getLogger(__name__)



def validate_paths(image_path: str, ner_model_path: str, cv_model_path: str) -> None:
    checks = {
        "image_path":     Path(image_path),
        "ner_model_path": Path(ner_model_path),
        "cv_model_path":  Path(cv_model_path),
    }
    for arg, path in checks.items():
        if not path.exists():
            logger.error(f"Path not found for --{arg}: '{path}'")
            sys.exit(1)


def run_pipeline(
    text: str,
    image_path: str,
    ner_pipeline,
    cv_model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
    min_confidence: float = 0.0,
) -> bool:
    """
    Execute the multimodal verification pipeline.

    Both models are passed in as pre-loaded objects so that this function
    can be called multiple times (e.g. in a notebook or batch loop) without
    reloading weights on every invocation.

    Args:
        text:           Free-form text describing an animal.
        image_path:     Path to the image to classify.
        ner_pipeline:   Pre-loaded HuggingFace NER pipeline.
        cv_model:       Pre-loaded ResNet-18 classifier.
        class_names:    Ordered list of class labels from the CV checkpoint.
        device:         Torch device for CV inference.
        min_confidence: Minimum NER confidence threshold.

    Returns:
        True if the CV-predicted animal matches any animal extracted from text.
    """
    extracted = extract_animal_names(text, ner_pipeline, min_confidence)
    logger.info(f"NER extraction  : {extracted if extracted else 'none detected'}")

    cv_results = predict_cv(image_path, cv_model, class_names, device, top_k=1)
    predicted_animal, confidence = cv_results[0]
    logger.info(f"CV prediction   : {predicted_animal}  ({confidence:.1f}%)")

    extracted_lower = [a.lower() for a in extracted]
    match = predicted_animal.lower() in extracted_lower

    return match



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Animal Verification Pipeline: NER (text) + ResNet-18 (image)."
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="Input text describing an animal.",
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to the image file to classify.",
    )
    parser.add_argument(
        "--ner_model_path", type=str, required=True,
        help="Path to the fine-tuned NER model directory.",
    )
    parser.add_argument(
        "--cv_model_path", type=str, required=True,
        help="Path to the CV model checkpoint (.pth).",
    )
    parser.add_argument(
        "--min_confidence", type=float, default=0.0,
        help="Minimum NER entity confidence to include (0.0–1.0, default: 0.0).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: 'cpu' or 'cuda'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    validate_paths(args.image_path, args.ner_model_path, args.cv_model_path)

    logger.info("Loading NER model...")
    ner_pipeline = load_ner_pipeline(
        args.ner_model_path,
        device=0 if device.type == "cuda" else -1,
    )

    logger.info("Loading CV model...")
    cv_model, class_names = load_cv_model(args.cv_model_path, device=device)

    result = run_pipeline(
        text           = args.text,
        image_path     = args.image_path,
        ner_pipeline   = ner_pipeline,
        cv_model       = cv_model,
        class_names    = class_names,
        device         = device,
        min_confidence = args.min_confidence,
    )

    separator = "=" * 45
    print(f"\n{separator}")
    print(f"  Text  : {args.text}")
    print(f"  Image : {args.image_path}")
    print(separator)
    print(f"  RESULT: {result}")
    print(f"{separator}\n")


if __name__ == "__main__":
    main()