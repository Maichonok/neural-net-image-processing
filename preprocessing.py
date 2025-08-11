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

# 3. Center crop the image to 224x224 pixels
center_crop = transforms.CenterCrop(224)
cropped_image = center_crop(resized_image)
print("Image size after CenterCrop:", cropped_image.size)

plt.imshow(cropped_image)
plt.title("Image after CenterCrop")
plt.axis("off")
plt.show()

# 4. Convert the cropped image to a tensor
to_tensor = transforms.ToTensor()
tensor_image = to_tensor(cropped_image)
print("Tensor size after ToTensor:", tensor_image.shape)  # Expected: torch.Size([3, 224, 224])