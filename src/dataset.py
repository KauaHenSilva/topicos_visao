import os
import sys
import zipfile
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    PATH_DOWNLOAD_DB, PATH_DB_BASE, TRAIN_CSV, TEST_CSV, VAL_CSV,
    TRAIN_DIR, TEST_DIR, VAL_DIR,
    IMG_SIZE_V1, BATCH_SIZE_V1,
    IMG_SIZE_V2, BATCH_SIZE_V2
)
from data_preprocessing import preprocess_dataset_cache

def load_and_balance_data():
    if not os.path.exists(TRAIN_CSV):
        if os.path.exists(PATH_DOWNLOAD_DB):
            print("Descompactando database localmente...")
            with zipfile.ZipFile(PATH_DOWNLOAD_DB, "r") as zip_ref:
                zip_ref.extractall(PATH_DB_BASE)
        else:
            print(f"Erro: Dataset não encontrado e o arquivo {PATH_DOWNLOAD_DB} também não existe.")
            sys.exit(1)

    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    df_val = pd.read_csv(VAL_CSV)

    for df in [df_train, df_test, df_val]:
        df["ID"] = df["ID"].astype(str)
        df["filename"] = df["ID"] + ".png"

    df_majority = df_train[df_train["Disease_Risk"] == 0]
    df_minority = df_train[df_train["Disease_Risk"] == 1]

    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_train_balanceado = pd.concat([df_majority, df_minority_upsampled])
    df_train_balanceado = df_train_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

    y_values = df_train_balanceado["Disease_Risk"].values
    pesos = compute_class_weight("balanced", classes=np.unique(y_values), y=y_values)
    pesos_dit = {int(k): float(v) for k, v in enumerate(pesos)}

    print(f"Total Treino Balanceado: {len(df_train_balanceado)} | Teste: {len(df_test)} | Validação: {len(df_val)}")
    
    preprocess_dataset_cache()
    
    return df_train_balanceado, df_test, df_val, pesos_dit

def get_data_generators(df_train_balanceado, df_val, df_test, version="V1"):
    aug_params = dict(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.15, horizontal_flip=True, vertical_flip=True,
        brightness_range=[0.8, 1.2], fill_mode="nearest"
    )

    datagen_aug = ImageDataGenerator(**aug_params)
    datagen_clean = ImageDataGenerator()

    if version == "V1":
        img_size = IMG_SIZE_V1
        batch_size = BATCH_SIZE_V1
    else:
        img_size = IMG_SIZE_V2
        batch_size = BATCH_SIZE_V2

    print(f"Configurando Geradores {version} (Resolução: {img_size}, Batch: {batch_size})...")
    
    train_gen = datagen_aug.flow_from_dataframe(
        dataframe=df_train_balanceado, directory=TRAIN_DIR, x_col="filename", y_col="Disease_Risk",
        class_mode="raw", target_size=img_size, batch_size=batch_size, shuffle=True
    )
    
    val_gen = datagen_clean.flow_from_dataframe(
        dataframe=df_val, directory=VAL_DIR, x_col="filename", y_col="Disease_Risk",
        class_mode="raw", target_size=img_size, batch_size=batch_size, shuffle=False
    )
    
    test_gen = datagen_clean.flow_from_dataframe(
        dataframe=df_test, directory=TEST_DIR, x_col="filename", y_col="Disease_Risk",
        class_mode="raw", target_size=img_size, batch_size=batch_size, shuffle=False
    )
    
    return train_gen, val_gen, test_gen
