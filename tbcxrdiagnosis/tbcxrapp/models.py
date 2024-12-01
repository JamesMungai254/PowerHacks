from django.db import models  # Assuming this is part of a Django app
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

# The model architecture
class TBModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TBModel, self).__init__()
        # Directly initialize the ResNet model
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
# Load the model
def load_model():
    model = TBModel(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the state dictionary and prepend 'resnet.' to keys
    state_dict = torch.load('tb_detection_model.pth', map_location=device)
    new_state_dict = {f"resnet.{k}": v for k, v in state_dict.items()}
    
    # Load the updated state dictionary
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


# Preprocessing the uploaded image
def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure consistent size
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for pre-trained models
    ])
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure RGB mode
    except FileNotFoundError:
        raise FileNotFoundError(f"The image file '{image_path}' was not found.")
    except Exception as e:
        raise ValueError(f"Error processing the image: {e}")
    
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Make prediction
def predict(image_path):
    model = load_model()  # Load the trained model
    image_tensor = process_image(image_path)  # Process the image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)  # Send image to the same device as the model
    with torch.no_grad():
        outputs = model(image_tensor)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get the class index with the highest probability
    return predicted.item()  # Return 0 for normal, 1 for tuberculosis
