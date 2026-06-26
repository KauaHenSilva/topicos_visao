import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import (
    IMG_SIZE_V1, CHECKPOINT_PATH_V1,
    IMG_SIZE_V2, CHECKPOINT_PATH_V2,
    LEARNING_RATE_BASE, LEARNING_RATE_FINE
)

def build_model(version="V1"):
    # IMPORTANTE: Evita colisão de nomes nas camadas do Keras quando 
    # instanciamos mais de um modelo sequencialmente na mesma sessão.
    # tf.keras.backend.clear_session()
    
    if version == "V1":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=IMG_SIZE_V1 + (3,))
    else:
        base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=IMG_SIZE_V2 + (3,))
        
    base_model.trainable = False # Congela a rede original

    suffix = "" if version == "V1" else "_1"
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(name=f"global_average_pooling2d{suffix}"),
        Dropout(0.2, name=f"dropout{suffix}"),
        Dense(1, activation="sigmoid", name=f"dense{suffix}")
    ])

    return model, base_model

def get_callbacks(version="V1"):
    checkpoint_path = CHECKPOINT_PATH_V1 if version == "V1" else CHECKPOINT_PATH_V2
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    ]

def compile_model_base(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_BASE),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, apply_class_balancing=False),
        metrics=["accuracy"]
    )
    return model

def compile_model_fine_tuning(model, base_model):
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, apply_class_balancing=False),
        metrics=["accuracy"]
    )
    return model
