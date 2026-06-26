import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from config import TEST_DIR_ORIGINAL, IMG_SIZE_V1, IMG_SIZE_V2, RESULTS_DIR, CHECKPOINT_PATH_V1, CHECKPOINT_PATH_V2, GRADCAM_DIR, RELATORIOS_DIR
from data_preprocessing import get_clahe_image_for_model
from models import build_model

def get_gradcam_heatmap_and_pred(model, img_rgb, img_size):
    model_input, original_cropped = get_clahe_image_for_model(img_rgb, img_size)
    arr = np.expand_dims(model_input, axis=0)

    grad_model = tf.keras.models.Model(
        model.layers[0].input,
        [model.layers[0].get_layer("top_conv").output, model.layers[0].output]
    )

    with tf.GradientTape() as tape:
        conv_out, base_out = grad_model(arr)
        x = model.layers[1](base_out)
        x = model.layers[2](x, training=False)
        pred = model.layers[3](x)
        loss = pred[:, 0]

    hm = tf.reduce_mean(tape.gradient(loss, conv_out), axis=(0, 1, 2))
    hm = tf.maximum(conv_out[0] @ hm[..., tf.newaxis], 0) / (tf.math.reduce_max(tf.maximum(conv_out[0] @ hm[..., tf.newaxis], 0)) + 1e-10)
    
    hm_resized = cv2.resize(hm.numpy().squeeze().astype(np.float32), (original_cropped.shape[1], original_cropped.shape[0]))
    hm_color = np.uint8(255 * hm_resized)
    
    return hm_resized, hm_color, pred.numpy()[0][0], original_cropped

def manual_load_weights(model, filepath):
    import h5py
    with h5py.File(filepath, 'r') as f:
        base_group = f['efficientnetb3']
        w_names = base_group.attrs.get('weight_names', [])
        saved_weights = {}
        for w in w_names:
            w_str = w.decode('utf8') if hasattr(w, 'decode') else w
            ds = base_group[w_str]
            saved_weights[w_str] = ds[()] if not ds.shape else ds[:]
            
        keras_w_to_set = []
        base_layer = model.layers[0]
        for w in base_layer.weights:
            clean_name = '/'.join(w.name.split('/')[1:])
            if clean_name in saved_weights:
                keras_w_to_set.append((w, saved_weights[clean_name]))
        tf.keras.backend.batch_set_value(keras_w_to_set)

        top_group = f['top_level_model_weights']
        top_names = top_group.attrs.get('weight_names', [])
        saved_top_weights = {}
        for w in top_names:
            w_str = w.decode('utf8') if hasattr(w, 'decode') else w
            ds = top_group[w_str]
            saved_top_weights[w_str] = ds[()] if not ds.shape else ds[:]

        keras_top_to_set = []
        for w in model.weights:
            if not any(w is bw for bw in base_layer.weights):
                clean_name = w.name
                if clean_name in saved_top_weights:
                    keras_top_to_set.append((w, saved_top_weights[clean_name]))
                else:
                    clean_name2 = clean_name.replace('_1/', '/')
                    if clean_name2 in saved_top_weights:
                        keras_top_to_set.append((w, saved_top_weights[clean_name2]))
        tf.keras.backend.batch_set_value(keras_top_to_set)

def generate_gradcam_comparative(df_test):
    print("\nGerando Grad-CAM Comparativo...")

    idx_doente = df_test[df_test['Disease_Risk'] == 1].index[0]
    img_path = os.path.join(TEST_DIR_ORIGINAL, df_test.iloc[idx_doente]["filename"])
    img_raw = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # --- PROCESSAMENTO V1 ---
    print("Processando Grad-CAM V1...")
    tf.keras.backend.clear_session()
    model_v1, _ = build_model(version="V1")
    try:
        model_v1.load_weights(CHECKPOINT_PATH_V1)
    except Exception as e:
        print(f"Fallback V1: {e}")
        model_v1.load_weights(CHECKPOINT_PATH_V1, by_name=True, skip_mismatch=True)
    hm_v1_resized, hm_v1_color, pred_v1, original_cropped = get_gradcam_heatmap_and_pred(model_v1, img_rgb, IMG_SIZE_V1)

    # --- PROCESSAMENTO V2 ---
    print("Processando Grad-CAM V2...")
    tf.keras.backend.clear_session()
    model_v2, _ = build_model(version="V2")
    try:
        model_v2.load_weights(CHECKPOINT_PATH_V2)
    except Exception as e:
        print(f"Fallback V2 manual loader (Keras 2 bug): {e}")
        manual_load_weights(model_v2, CHECKPOINT_PATH_V2)
    hm_v2_resized, hm_v2_color, pred_v2, _ = get_gradcam_heatmap_and_pred(model_v2, img_rgb, IMG_SIZE_V2)

    # --- PLOTAGEM COMPARATIVA ---
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]

    gray = cv2.cvtColor(original_cropped, cv2.COLOR_RGB2GRAY)
    _, mask_eye = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_eye, [cv2.convexHull(c)], -1, 255, thickness=cv2.FILLED)
    mask_bool = mask_eye > 0

    sup_v1 = np.zeros_like(original_cropped, dtype=np.float32)
    alpha_v1 = 0.6 * hm_v1_resized[mask_bool, np.newaxis]
    sup_v1[mask_bool] = np.clip(jet_colors[hm_v1_color[mask_bool]] * alpha_v1 + (original_cropped[mask_bool] / 255.0) * (1 - alpha_v1), 0.0, 1.0)

    sup_v2 = np.zeros_like(original_cropped, dtype=np.float32)
    alpha_v2 = 0.6 * hm_v2_resized[mask_bool, np.newaxis]
    sup_v2[mask_bool] = np.clip(jet_colors[hm_v2_color[mask_bool]] * alpha_v2 + (original_cropped[mask_bool] / 255.0) * (1 - alpha_v2), 0.0, 1.0)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_cropped)
    plt.title(f"Imagem Base (Proporção Original)\nRisco Real: {df_test.iloc[idx_doente]['Disease_Risk']}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sup_v1)
    plt.title(f"Grad-CAM V1 (EfficientNetB0)\nPred: {pred_v1:.3f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(sup_v2)
    plt.title(f"Grad-CAM V2 (EfficientNetB3)\nPred: {pred_v2:.3f}")
    plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join(GRADCAM_DIR, "comparativo_v1_v2.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM Comparativo salvo em: {out_path}")


def generate_categorized_gradcams(df_test, csv_path=None, num_samples=2):
    import pandas as pd
    if csv_path is None:
        csv_path = os.path.join(RELATORIOS_DIR, "detalhamento_predicoes_v1_v2.csv")

    if not os.path.exists(csv_path):
        print(f"Erro: O arquivo {csv_path} não existe. Você deve executar a avaliação primeiro (evaluate_models_comparative).")
        return
    
    df_preds = pd.read_csv(csv_path)
    
    cond_acertou_v1 = df_preds['Pred_V1'] == df_preds['Rotulo_Real']
    cond_acertou_v2 = df_preds['Pred_V2'] == df_preds['Rotulo_Real']

    df_ambos_acertaram = df_preds[cond_acertou_v1 & cond_acertou_v2]
    df_ambos_erraram = df_preds[(~cond_acertou_v1) & (~cond_acertou_v2)]
    df_v1_acertou_v2_errou = df_preds[cond_acertou_v1 & (~cond_acertou_v2)]
    df_v1_errou_v2_acertou = df_preds[(~cond_acertou_v1) & cond_acertou_v2]

    categories = {
        "Ambos_Acertaram": df_ambos_acertaram,
        "Ambos_Erraram": df_ambos_erraram,
        "V1_Acertou_V2_Errou": df_v1_acertou_v2_errou,
        "V1_Errou_V2_Acertou": df_v1_errou_v2_acertou
    }

    selected_images = []
    for cat_name, df_cat in categories.items():
        n = min(num_samples, len(df_cat))
        sampled = df_cat.sample(n, random_state=42) if n > 0 else df_cat
        for _, row in sampled.iterrows():
            selected_images.append({
                "category": cat_name,
                "filename": row['Filename'],
                "real": row['Rotulo_Real'],
                "pred_v1": row['Pred_V1'],
                "pred_v2": row['Pred_V2'],
                "prob_v1": row['Prob_V1'],
                "prob_v2": row['Prob_V2']
            })

    if not selected_images:
        print("Nenhuma imagem classificada nas predições (o CSV está vazio?).")
        return

    print(f"Calculando heatmaps V1 para {len(selected_images)} imagens...")
    tf.keras.backend.clear_session()
    model_v1, _ = build_model(version="V1")
    try:
        model_v1.load_weights(CHECKPOINT_PATH_V1)
    except Exception as e:
        model_v1.load_weights(CHECKPOINT_PATH_V1, by_name=True, skip_mismatch=True)
    
    v1_results = {}
    for item in selected_images:
        img_path = os.path.join(TEST_DIR_ORIGINAL, item["filename"])
        img_raw = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        hm_resized, hm_color, _, original_cropped = get_gradcam_heatmap_and_pred(model_v1, img_rgb, IMG_SIZE_V1)
        v1_results[item["filename"]] = (hm_resized, hm_color, original_cropped)

    print("Calculando heatmaps V2...")
    tf.keras.backend.clear_session()
    model_v2, _ = build_model(version="V2")
    try:
        model_v2.load_weights(CHECKPOINT_PATH_V2)
    except Exception as e:
        manual_load_weights(model_v2, CHECKPOINT_PATH_V2)
    
    v2_results = {}
    for item in selected_images:
        img_path = os.path.join(TEST_DIR_ORIGINAL, item["filename"])
        img_raw = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        hm_resized, hm_color, _, _ = get_gradcam_heatmap_and_pred(model_v2, img_rgb, IMG_SIZE_V2)
        v2_results[item["filename"]] = (hm_resized, hm_color)

    print("Gerando visualizações...")
    base_out_dir = os.path.join(GRADCAM_DIR, "gradcam_categories")
    os.makedirs(base_out_dir, exist_ok=True)
    
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]

    for item in selected_images:
        cat_dir = os.path.join(base_out_dir, item["category"])
        os.makedirs(cat_dir, exist_ok=True)
        
        fname = item["filename"]
        hm_v1_resized, hm_v1_color, original_cropped = v1_results[fname]
        hm_v2_resized, hm_v2_color = v2_results[fname]

        gray = cv2.cvtColor(original_cropped, cv2.COLOR_RGB2GRAY)
        _, mask_eye = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_eye, [cv2.convexHull(c)], -1, 255, thickness=cv2.FILLED)
        mask_bool = mask_eye > 0

        sup_v1 = np.zeros_like(original_cropped, dtype=np.float32)
        alpha_v1 = 0.6 * hm_v1_resized[mask_bool, np.newaxis]
        sup_v1[mask_bool] = np.clip(jet_colors[hm_v1_color[mask_bool]] * alpha_v1 + (original_cropped[mask_bool] / 255.0) * (1 - alpha_v1), 0.0, 1.0)

        sup_v2 = np.zeros_like(original_cropped, dtype=np.float32)
        alpha_v2 = 0.6 * hm_v2_resized[mask_bool, np.newaxis]
        sup_v2[mask_bool] = np.clip(jet_colors[hm_v2_color[mask_bool]] * alpha_v2 + (original_cropped[mask_bool] / 255.0) * (1 - alpha_v2), 0.0, 1.0)

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(original_cropped)
        plt.title(f"Base | Rótulo Real: {item['real']}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(sup_v1)
        plt.title(f"V1 | Pred: {item['pred_v1']} (Prob: {item['prob_v1']:.3f})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(sup_v2)
        plt.title(f"V2 | Pred: {item['pred_v2']} (Prob: {item['prob_v2']:.3f})")
        plt.axis("off")

        plt.tight_layout()
        out_name = fname.replace('/', '_').replace('\\', '_')
        if not out_name.endswith('.png'): out_name += '.png'
        out_path = os.path.join(cat_dir, out_name)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"Sucesso! {len(selected_images)} imagens foram organizadas em '{base_out_dir}'.")
