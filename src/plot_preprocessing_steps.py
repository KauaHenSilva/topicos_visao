import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import TEST_DIR_ORIGINAL, GRAFICOS_DIR

def plot_preprocessing_steps(img_filename=None):
    if img_filename is None:
        # Pega a primeira imagem do teste
        files = [f for f in os.listdir(TEST_DIR_ORIGINAL) if f.endswith('.png')]
        if not files:
            print("Nenhuma imagem encontrada.")
            return
        img_filename = os.path.join(TEST_DIR_ORIGINAL, files[0])
    
    print(f"Processando a imagem: {img_filename}")
    img_bgr = cv2.imread(img_filename)
    if img_bgr is None:
        print("Erro ao ler imagem.")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Passo 1: Original
    img_uint8 = img_rgb.copy()
    
    # Passo 2: Grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # Passo 3: Threshold / Mask
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Passo 4: Bounding Box (Crop)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, img_uint8.shape[1], img_uint8.shape[0]
    
    crop_vis = img_uint8.copy()
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(crop_vis, (x, y), (x+w, y+h), (255, 0, 0), 10)
        
    # Aplicando o crop
    img_cropped = img_uint8[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]
    
    # Passo 5: CLAHE
    lab = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Plotando tudo (4 passos essenciais)
    # Usa gridspec_kw para que a largura dos subplots seja proporcional à largura real das imagens
    w_orig = img_rgb.shape[1]
    w_crop = img_cropped.shape[1]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={'width_ratios': [w_orig, w_orig, w_crop, w_crop]})
    axes = axes.flatten()
    
    steps = [
        ("1. Imagem Original", img_rgb, None),
        ("2. Localização (Crop)", crop_vis, None),
        ("3. Imagem Recortada", img_cropped, None),
        ("4. Realce (CLAHE)", clahe_img, None)
    ]
    
    for ax, (title, img, cmap) in zip(axes, steps):
        if cmap is None and len(img.shape) == 2:
            cmap = 'gray'
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(GRAFICOS_DIR, "passo_a_passo_clahe.png")
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo em: {out_path}")

if __name__ == '__main__':
    plot_preprocessing_steps()
