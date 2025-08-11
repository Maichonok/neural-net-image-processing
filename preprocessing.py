from torchvision import transforms
from PIL import Image

# Create a transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),          # Resize: smaller side to 256 pixels
    transforms.CenterCrop(224),      # Center crop to 224x224
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(            # Normalize using ImageNet statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the image and convert to RGB (in case the image has an alpha channel)
image = Image.open("./data/sneakers.jpg").convert("RGB")

# Apply the transformation pipeline
tensor_image = transform(image)

# Print tensor size to verify
print("Tensor size:", tensor_image.shape)  # Should print torch.Size([3, 224, 224])