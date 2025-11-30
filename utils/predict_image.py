import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def load_model(model_path, num_classes):
    model = models.resnet18(weights=None) # No weights needed, we load state_dict
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image, model_type):
    """
    Predicts class for an image.
    image: PIL Image
    model_type: 'brain_tumor' or 'skin_cancer'
    """
    if model_type == 'brain_tumor':
        model_path = "models/brain_tumor_model.pth"
        classes_path = "models/brain_tumor_classes.txt"
        num_classes = 4
    elif model_type == 'skin_cancer':
        model_path = "models/skin_cancer_model.pth"
        classes_path = "models/skin_cancer_classes.txt"
        num_classes = 9
    else:
        return None, "Invalid model type"
        
    if not os.path.exists(model_path):
        return None, "Model not found"

    # Load classes
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Transforms (Must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image).unsqueeze(0)
    
    # Load model
    model = load_model(model_path, num_classes)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, preds = torch.max(outputs, 1)
        
    predicted_class = class_names[preds[0]]
    confidence = probabilities[preds[0]].item()
    
    return predicted_class, confidence
