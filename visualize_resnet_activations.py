from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch
from matplotlib import pyplot as plt

# Load pretrained ResNet18 model and its preprocessing transforms
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

image_path = "./data/sneakers.jpg"
image = Image.open(image_path).convert('RGB')

# Define preprocessing pipeline: resize → crop → tensor → normalize
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Apply preprocessing and add batch dimension
input_tensor = preprocess(image).unsqueeze(0)

# Function to visualize layer activations
def visualize_layer_activations(model, image_tensor, layer_name='layer1'):
    activations = {}

    # Hook function to store the output of the selected layer
    def hook_fn(module, input, output):
        activations[layer_name] = output

    # Register the hook for the selected layer
    if layer_name == 'layer1':
        model.layer1.register_forward_hook(hook_fn)

    # Forward pass without computing gradients
    with torch.no_grad():
        _ = model(image_tensor)

    # Extract the activations from the hook
    layer_activation = activations[layer_name][0].cpu().numpy()

    # Plot the first 16 activation maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < layer_activation.shape[0]:
            ax.imshow(layer_activation[i], cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Activations of layer {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Visualize activations for an early layer (basic features)
visualize_layer_activations(model, input_tensor, 'layer1')
