import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TEST_DIR_ORIGINAL, GRAFICOS_DIR
from data_preprocessing import apply_clahe

def plot_augmentation_steps(img_filename=None):
    if img_filename is None:
        files = [f for f in os.listdir(TEST_DIR_ORIGINAL) if f.endswith('.png')]
        img_filename = os.path.join(TEST_DIR_ORIGINAL, files[0])
    
    print(f"Gerando exemplos de augmentation para: {img_filename}")
    img_bgr = cv2.imread(img_filename)
    if img_bgr is None:
        print("Erro ao ler imagem.")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Pre-processamento para obter a imagem final do CLAHE
    clahe_img = apply_clahe(img_rgb)
    # Redimensionamento para o tamanho usado pela rede (384x384)
    clahe_img_resized = cv2.resize(clahe_img, (384, 384))
    
    img_batch = np.expand_dims(clahe_img_resized, 0)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes = axes.flatten()
    
    # 1. Original (Pós-CLAHE)
    axes[0].imshow(clahe_img_resized)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis('off')
    
    # 2. Rotação (50 graus)
    datagen_rot = ImageDataGenerator(rotation_range=50, fill_mode="nearest")
    aug_img_rot = next(datagen_rot.flow(img_batch, batch_size=1, seed=42))[0].astype(np.uint8)
    axes[1].imshow(aug_img_rot)
    axes[1].set_title("Rotação (50°)", fontsize=14)
    axes[1].axis('off')
    
    # 3. Espelhamento Horizontal
    aug_img_flip_h = np.fliplr(clahe_img_resized)
    axes[2].imshow(aug_img_flip_h)
    axes[2].set_title("Espelhamento Horiz.", fontsize=14)
    axes[2].axis('off')
    
    # 4. Espelhamento Vertical
    aug_img_flip_v = np.flipud(clahe_img_resized)
    axes[3].imshow(aug_img_flip_v)
    axes[3].set_title("Espelhamento Vert.", fontsize=14)
    axes[3].axis('off')
    
    # 5. Brilho
    datagen_bright = ImageDataGenerator(brightness_range=[1.3, 1.3], fill_mode="nearest")
    aug_img_bright = next(datagen_bright.flow(img_batch, batch_size=1, seed=42))[0].astype(np.uint8)
    axes[4].imshow(aug_img_bright)
    axes[4].set_title("Brilho", fontsize=14)
    axes[4].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(GRAFICOS_DIR, "exemplos_data_augmentation.png")
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de Data Augmentation salvo em: {out_path}")

if __name__ == '__main__':
    plot_augmentation_steps()
