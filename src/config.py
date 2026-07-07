import os
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Caminhos base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_DB_BASE = os.path.join(BASE_DIR, "notebooks", "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "notebooks", "resultados_modelo")
MODELOS_DIR = os.path.join(RESULTS_DIR, "modelos")
GRAFICOS_DIR = os.path.join(RESULTS_DIR, "graficos")
RELATORIOS_DIR = os.path.join(RESULTS_DIR, "relatorios")
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")

# Garantir que os diretórios de resultados existam
for d in [MODELOS_DIR, GRAFICOS_DIR, RELATORIOS_DIR, GRADCAM_DIR]:
    os.makedirs(d, exist_ok=True)

PATH_DB_PREPROCESSED = os.path.join(BASE_DIR, "notebooks", "dataset_preprocessed")

# Diretórios Originais
TRAIN_DIR_ORIGINAL = os.path.join(PATH_DB_BASE, "Training_Set", "Training_Set", "Training")
TEST_DIR_ORIGINAL = os.path.join(PATH_DB_BASE, "Test_Set", "Test_Set", "Test")
VAL_DIR_ORIGINAL = os.path.join(PATH_DB_BASE, "Evaluation_Set", "Evaluation_Set", "Validation")

# Diretórios Pré-processados (Cache)
TRAIN_DIR = os.path.join(PATH_DB_PREPROCESSED, "Training")
TEST_DIR = os.path.join(PATH_DB_PREPROCESSED, "Test")
VAL_DIR = os.path.join(PATH_DB_PREPROCESSED, "Validation")

# CSVs de Rótulos
TRAIN_CSV = os.path.join(PATH_DB_BASE, "Training_Set", "Training_Set", "RFMiD_Training_Labels.csv")
TEST_CSV = os.path.join(PATH_DB_BASE, "Test_Set", "Test_Set", "RFMiD_Testing_Labels.csv")
VAL_CSV = os.path.join(PATH_DB_BASE, "Evaluation_Set", "Evaluation_Set", "RFMiD_Validation_Labels.csv")

# Download path
PATH_DOWNLOAD_DB = os.path.join(PATH_DB_BASE, "retinal-disease-classification.zip")

# Hiperparâmetros
EPOCAS_BASE = 10
EPOCAS_FINETUNE = 10
CLINICAL_THRESHOLD = 0.35
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_FINE = 1e-5

# Configurações V1 (EfficientNetB0) 
IMG_SIZE_V1 = (384, 384)
BATCH_SIZE_V1 = 2
CHECKPOINT_PATH_V1 = os.path.join(MODELOS_DIR, "melhor_modelo_v1.weights.h5")

# Configurações V2 (EfficientNetB3)
IMG_SIZE_V2 = (384, 384)
BATCH_SIZE_V2 = 2
CHECKPOINT_PATH_V2 = os.path.join(MODELOS_DIR, "melhor_modelo_v2.weights.h5")
