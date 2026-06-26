import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from config import TRAIN_DIR_ORIGINAL, TEST_DIR_ORIGINAL, VAL_DIR_ORIGINAL, TRAIN_DIR, TEST_DIR, VAL_DIR

def apply_clahe(img):
    """
    Localiza a retina, faz o crop das bordas pretas mortas, aplica CLAHE.
    Retorna a imagem processada mantendo sua proporção original,
    para que o redimensionamento ocorra isoladamente onde for necessário.
    """
    img_uint8 = (img * 255).astype(np.uint8) if img.dtype != np.uint8 and img.max() <= 1.0 else img.astype(np.uint8)

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, img_uint8.shape[1], img_uint8.shape[0]
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        img_uint8 = img_uint8[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    final_img = cv2.bitwise_and(final_img, final_img, mask=mask)
    return final_img

def pad_resize(img, target_size):
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    xoff = (target_size[0] - nw) // 2
    yoff = (target_size[1] - nh) // 2
    padded[yoff:yoff+nh, xoff:xoff+nw] = resized
    return padded

def get_clahe_image_for_model(img, target_size):
    """
    Retorna a imagem pronta para entrar na CNN (model_input)
    e a imagem recortada com CLAHE em tamanho original (original_cropped)
    """
    final_img = apply_clahe(img)
    # Importante: Como o usuário relatou achatamento da proporção, mantivemos cv2.resize puro na Pipeline
    # Se for necessário letterboxing (pad_resize), isso deve ser trocado aqui e re-treinado o modelo.
    # Como re-treinamos com cv2.resize(..., target_size) para preencher os 384x384 integralmente,
    # continuamos usando o resize normal.
    model_input = cv2.resize(final_img, target_size)
    return model_input, final_img

def process_and_save_image(src_path, dst_path):
    if os.path.exists(dst_path):
        return
    img = cv2.imread(src_path)
    if img is None:
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_clahe = apply_clahe(img_rgb)
    cv2.imwrite(dst_path, cv2.cvtColor(img_clahe.astype(np.uint8), cv2.COLOR_RGB2BGR))

def process_and_save_image_unpack(args):
    return process_and_save_image(*args)

def preprocess_dataset_cache():
    print("Verificando a necessidade de pré-processar as imagens para cache...")
    
    tasks = []
    os.makedirs(TRAIN_DIR, exist_ok=True)
    if os.path.exists(TRAIN_DIR_ORIGINAL):
        for f in os.listdir(TRAIN_DIR_ORIGINAL):
            if f.endswith('.png'):
                tasks.append((os.path.join(TRAIN_DIR_ORIGINAL, f), os.path.join(TRAIN_DIR, f)))
                
    os.makedirs(TEST_DIR, exist_ok=True)
    if os.path.exists(TEST_DIR_ORIGINAL):
        for f in os.listdir(TEST_DIR_ORIGINAL):
            if f.endswith('.png'):
                tasks.append((os.path.join(TEST_DIR_ORIGINAL, f), os.path.join(TEST_DIR, f)))
                
    os.makedirs(VAL_DIR, exist_ok=True)
    if os.path.exists(VAL_DIR_ORIGINAL):
        for f in os.listdir(VAL_DIR_ORIGINAL):
            if f.endswith('.png'):
                tasks.append((os.path.join(VAL_DIR_ORIGINAL, f), os.path.join(VAL_DIR, f)))
            
    tasks = [t for t in tasks if not os.path.exists(t[1])]
    
    if tasks:
        print(f"Pré-processando {len(tasks)} imagens. Isso pode levar alguns minutos (executado apenas uma vez)...")
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for i, _ in enumerate(executor.map(process_and_save_image_unpack, tasks)):
                if (i + 1) % 100 == 0:
                    print(f"Processado {i + 1}/{len(tasks)} imagens...")
        print("Pré-processamento concluído!")
    else:
        print("Imagens já estão pré-processadas.")
