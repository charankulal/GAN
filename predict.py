import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from .text_encoder import TextEncoder
from .generator import Generator
from .discriminator import Discriminator

class Predict:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = "C:\\Users\\Public\\Documents\\Text-Image-GAN\\app\\gan_models\\checkpoint.pth"
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Initialize models
        self.models = {
            "text_encoder": TextEncoder().to(self.device),
            "generator": Generator().to(self.device),
            "discriminator": Discriminator().to(self.device)
        }
        
        # Load checkpoint
        for key in self.models:
            self.models[key].load_state_dict(self.checkpoint['models'][key])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def save_image(self, image_tensor, file_name, height=64, width=64):
        image = image_tensor.clamp(0, 1)
        image = image.permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

        file_path = os.path.join("static", "generated_images", file_name)
        pil_image.save(file_path)
        return file_path 
    def visualize_and_save(self, image, text, title=None, file_path=None):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        if title:
            ax.set_title(title)
        
        if self.tokenizer:
            decoded_text = self.tokenizer.decode(text['input_ids'][0].tolist(), skip_special_tokens=True)
            ax.set_xlabel(decoded_text)
        
        if file_path:
            self.save_image(image, file_path, height=128, width=128)
            print(f"Image saved as {file_path}")
