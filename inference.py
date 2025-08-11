import torch
from torchvision import models, transforms
from PIL import Image
import requests

# Load ImageNet class labels from a URL
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
classes = requests.get(imagenet_labels_url).json()

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Set the model to evaluation mode (inference)
model.eval()

# Create the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),            # Resize smaller side to 256 pixels
    transforms.CenterCrop(224),        # Center crop to 224x224 pixels
    transforms.ToTensor(),             # Convert image to PyTorch tensor
    transforms.Normalize(              # Normalize with ImageNet statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load and preprocess the image
image = Image.open("./data/ball.jpg").convert('RGB')
tensor_image = transform(image)

# Add batch dimension: model expects input shape [batch_size, channels, height, width]
image_tensor = tensor_image.unsqueeze(0)

# Perform inference without computing gradients
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted_idx = torch.max(outputs, 1)               # Get the index of the predicted class
    predicted_label = classes[predicted_idx.item()]         # Map to human-readable label

print(f"Predicted class: {predicted_label}")
