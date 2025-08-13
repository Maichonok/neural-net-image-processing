## Key Stages of Image Preprocessing for Neural Networks

`Resize` → `CenterCrop` → `ToTensor` → `Normalize`

This project uses the **ResNet** architecture as the neural network model.

---

## Additional Highlights of This Project

- Implemented extraction of **top‑3 predictions** using `torch.topk`.
- Conducted experiments to assess how **noise, blurring, and imperfect cropping** affect the model's prediction confidence and probability distribution.
- Demonstrated the importance of **high‑quality image preprocessing** and why ResNet is sensitive to image details.
- Explored methods to **improve robustness** and classification accuracy for new or domain‑specific images.
