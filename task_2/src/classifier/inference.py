import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

def load_model(model_path, num_classes=10, device='cpu'):
    """Instantiates the model architecture and loads the trained weights."""
    model = models.resnet18(weights=None)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    return model.to(device)

def predict(image_path, model, device):
    """Processes an image and returns the predicted class name."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Ensure RGB format for consistency in case of grayscale inputs
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms and add batch dimension (1, C, H, W)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)

    return CLASSES[predicted_idx.item()]

def main():
    parser = argparse.ArgumentParser(description="Test the Animals-10 classifier on a single image")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the best_animal_classifier.pth')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file to classify')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(args.model_path, num_classes=10, device=device)
    animal_name = predict(args.image_path, model, device)
    
    print(f"Predicted animal: {animal_name.upper()}")

if __name__ == '__main__':
    main()