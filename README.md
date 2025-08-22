## Overview

The initial layers of ResNet extract low-level features such as edges, textures, and colors. These features serve as the foundational building blocks for image understanding. As the network deepens, subsequent layers progressively combine and abstract these low-level features into high-level semantic representations that reflect the meaning and composition of whole objects within the image. This hierarchical feature extraction is fundamental to the success of deep convolutional neural networks like ResNet in visual recognition tasks.

## Key Stages of Image Preprocessing for Neural Networks

`Resize` → `CenterCrop` → `ToTensor` → `Normalize`

This project uses the **ResNet** architecture as the neural network model.

---

## Additional Highlights of This Project

- Implemented extraction of **top‑3 predictions** using `torch.topk`.
- Conducted experiments to assess how **noise, blurring, and imperfect cropping** affect the model's prediction confidence and probability distribution.
- Demonstrated the importance of **high‑quality image preprocessing** and why ResNet is sensitive to image details.
- Explored methods to **improve robustness** and classification accuracy for new or domain‑specific images.

## Sentiment Prediction in Action

| Class | Sentiment |
| ----- | --------- |
| 0     | NEUTRAL   |
| 1     | POSITIVE  |
| 2     | NEGATIVE  |
