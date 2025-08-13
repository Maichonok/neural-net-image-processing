import torch
from torchvision import models, transforms
from PIL import Image, ImageFilter
import requests
import matplotlib.pyplot as plt

# Load ImageNet class labels from a URL (JSON format)
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
classes = requests.get(imagenet_labels_url).json()

# Initialize an empty ResNet18 model (without pretrained weights)
model = models.resnet18(weights=None)
# Load pretrained weights from a local file
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Set the model to evaluation mode for inference (disables dropout, batchnorm updates, etc.)
model.eval()

# Define the image transformation pipeline to prepare input images for the model
transform = transforms.Compose([
    transforms.Resize(256),                  # Resize the shorter side of the image to 256 pixels
    transforms.CenterCrop(224),              # Crop the center 224x224 pixels from the resized image
    transforms.ToTensor(),                   # Convert the image to a PyTorch tensor (C x H x W, range [0,1])
    transforms.Normalize(                    # Normalize the tensor using ImageNet mean and std values
        [0.485, 0.456, 0.406],              # Mean for each channel (R, G, B)
        [0.229, 0.224, 0.225]               # Standard deviation for each channel (R, G, B)
    )
])

def predict_top3(image):
    # Apply the transformations to the input image and add a batch dimension (N=1)
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():  # Disable gradient calculation for inference efficiency
        outputs = model(image_tensor)  # Forward pass: get raw model outputs (logits)
    
    # Convert logits to probabilities using softmax on the class dimension
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Extract top-3 probabilities and their corresponding class indices
    top3_probs, top3_indices = torch.topk(probabilities, 3)
    
    results = []
    for i in range(3):
        # Map each index to the class label from ImageNet labels
        label = classes[top3_indices[i].item()]
        # Convert probability to percentage
        prob = top3_probs[i].item() * 100
        results.append(f"{label}: {prob:.2f}%")
    
    return results

def crop_corner(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
    cropped = image.crop((image.width - 224, image.height - 224, image.width, image.height))
    return cropped

# Crop the image and perform prediction on the cropped part
cropped = crop_corner("./data/sneakers.jpg")
predictions = predict_top3(cropped)
print("Object in the corner:", predictions)

# Display the cropped image with a title, hide axis ticks
import matplotlib.pyplot as plt
plt.imshow(cropped)
plt.title("Cropped image (bottom-right corner 224x224)")
plt.axis("off")
plt.show()