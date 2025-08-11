from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Reading image
image = Image.open("./data/sneakers.jpg").convert("RGB")
resize_transform = transforms.Resize(256)
resized_image = resize_transform(image)
center_crop = transforms.CenterCrop(224)
cropped_image = center_crop(resized_image)
to_tensor = transforms.ToTensor()
tensor_image = to_tensor(cropped_image)

# Normalize the tensor with ImageNet statistics
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
normalized_tensor = normalize(tensor_image)
print("Normalization completed successfully!")