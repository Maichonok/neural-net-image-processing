from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Reading image
image = Image.open("./data/sneakers.jpg").convert("RGB")
print("Original image size:", image.size)

# For visualization: displaying original image
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")  # turn off axes for clarity
plt.show()