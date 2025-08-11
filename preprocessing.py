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

# 2. Resize the image so that the smaller side is 256 pixels
resize_transform = transforms.Resize(256)
resized_image = resize_transform(image)
print("Image size after Resize:", resized_image.size)

plt.imshow(resized_image)
plt.title("Image after Resize")
plt.axis("off")
plt.show()