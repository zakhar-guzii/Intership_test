import argparse
import logging
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier.inference import load_model as load_cv_model
from src.classifier.inference import predict as predict_cv
from src.ner.inference import extract_animals

logger = logging.getLogger(__name__)

def run_pipeline(text: str, image_path: str, ner_model_path: str, cv_model_path: str, device: torch.device) -> bool:
    """Executes the multimodal pipeline comparing NER text entities with CV image predictions."""
    
    extracted_animals = extract_animals(text, ner_model_path)
    extracted_animals_lower = [animal.lower() for animal in extracted_animals]

    cv_model = load_cv_model(cv_model_path, num_classes=10, device=device)
    predicted_animal = predict_cv(image_path, cv_model, device)
    predicted_animal_lower = predicted_animal.lower()

    logger.info(f"NER Extraction: {extracted_animals if extracted_animals else 'None'}")
    logger.info(f"CV Prediction: {predicted_animal}")

    return predicted_animal_lower in extracted_animals_lower

def main():
    parser = argparse.ArgumentParser(description="Multimodal Pipeline: Text (NER) + Image (Classification)")
    parser.add_argument('--text', type=str, required=True, help='Input text description')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    parser.add_argument('--ner_model_path', type=str, required=True, help='Path to the fine-tuned NER model directory')
    parser.add_argument('--cv_model_path', type=str, required=True, help='Path to the CV model weights (.pth)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing pipeline on device: {device}")

    result = run_pipeline(
        text=args.text, 
        image_path=args.image_path, 
        ner_model_path=args.ner_model_path, 
        cv_model_path=args.cv_model_path, 
        device=device
    )
    
    logger.info(f"Pipeline Result: {result}")

if __name__ == '__main__':
    main()