import argparse

from transformers import pipeline

def extract_animals(text, model_path):
    """
    Loads the fine-tuned NER pipeline and extracts entities labeled as ANIMAL.
    """
    ner_pipeline = pipeline(
        "ner", 
        model=model_path, 
        tokenizer=model_path, 
        aggregation_strategy="simple"
    )
    
    predictions = ner_pipeline(text)
    
    extracted_animals = []
    for entity in predictions:
        if 'ANIMAL' in entity['entity_group']:
            extracted_animals.append(entity['word'].strip())
            
    return extracted_animals

def main():
    parser = argparse.ArgumentParser(description="Extract animal names from text using fine-tuned NER")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned NER model directory')
    parser.add_argument('--text', type=str, required=True, help='Input text to analyze')
    
    args = parser.parse_args()

    animals = extract_animals(args.text, args.model_path)
    
    print(f"📝 Input text: '{args.text}'")
    if animals:
        print(f" Found animals: {animals}")
    else:
        print("No animals found in the text.")

if __name__ == '__main__':
    main()