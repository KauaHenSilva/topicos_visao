import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR, EPOCAS_BASE, EPOCAS_FINETUNE, GRAFICOS_DIR, RELATORIOS_DIR
from models import build_model, compile_model_base, compile_model_fine_tuning, get_callbacks

def plot_training_history(history_base, history_fine, version="V1"):
    acc = history_base.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_base.history['val_accuracy'] + history_fine.history['val_accuracy']

    loss = history_base.history['loss'] + history_fine.history['loss']
    val_loss = history_base.history['val_loss'] + history_fine.history['val_loss']

    pd.DataFrame({
        'epoch': range(1, len(acc) + 1),
        'accuracy': acc,
        'val_accuracy': val_acc,
        'loss': loss,
        'val_loss': val_loss
    }).to_csv(os.path.join(RELATORIOS_DIR, f"historico_evolucao_{version.lower()}.csv"), index=False)

    transicao_idx = len(history_base.history['accuracy'])

    epochs_base = list(range(1, transicao_idx + 1))
    epochs_fine = list(range(1, len(acc) - transicao_idx + 1))

    # ==========================
    # GRÁFICO DA FASE BASE
    # ==========================
    plt.figure(figsize=(14, 6))

    # --- Gráfico de Acurácia (Base) ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_base, acc[:transicao_idx], label='Treino', linewidth=2, color='#1f77b4')
    plt.plot(epochs_base, val_acc[:transicao_idx], label='Validação', linewidth=2, color='#ff7f0e')
    plt.title(f'Evolução da Acurácia - {version} (Base)', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # --- Gráfico de Loss (Base) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_base, loss[:transicao_idx], label='Treino', linewidth=2, color='#1f77b4')
    plt.plot(epochs_base, val_loss[:transicao_idx], label='Validação', linewidth=2, color='#ff7f0e')
    plt.title(f'Evolução do Loss - {version} (Base)', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, f"evolucao_treinamento_{version.lower()}_base.png"))
    plt.close()

    # ==========================
    # GRÁFICO DA FASE FINE-TUNE
    # ==========================
    if len(epochs_fine) > 0:
        plt.figure(figsize=(14, 6))

        # --- Gráfico de Acurácia (Fine-Tune) ---
        plt.subplot(1, 2, 1)
        plt.plot(epochs_fine, acc[transicao_idx:], label='Treino', linewidth=2, color='#1f77b4')
        plt.plot(epochs_fine, val_acc[transicao_idx:], label='Validação', linewidth=2, color='#ff7f0e')
        plt.title(f'Evolução da Acurácia - {version} (Fine-Tune)', fontsize=14)
        plt.xlabel('Épocas', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

        # --- Gráfico de Loss (Fine-Tune) ---
        plt.subplot(1, 2, 2)
        plt.plot(epochs_fine, loss[transicao_idx:], label='Treino', linewidth=2, color='#1f77b4')
        plt.plot(epochs_fine, val_loss[transicao_idx:], label='Validação', linewidth=2, color='#ff7f0e')
        plt.title(f'Evolução do Loss - {version} (Fine-Tune)', fontsize=14)
        plt.xlabel('Épocas', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(GRAFICOS_DIR, f"evolucao_treinamento_{version.lower()}_finetune.png"))
        plt.close()

def train_pipeline(train_gen, val_gen, pesos_dit, version="V1"):
    print(f"\n--- INICIANDO PIPELINE {version} ---")
    model, base_model = build_model(version)
    model = compile_model_base(model)
    callbacks = get_callbacks(version)

    print(f"Treinando {version} Base (Camadas Congeladas)...")
    start_time = time.time()
    history_base = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCAS_BASE,
        class_weight=pesos_dit, callbacks=callbacks, verbose=1
    )

    print(f"\nDescongelando para Fine-Tuning {version}...")
    model = compile_model_fine_tuning(model, base_model)

    history_fine = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCAS_BASE + EPOCAS_FINETUNE,
        initial_epoch=history_base.epoch[-1] + 1, class_weight=pesos_dit,
        callbacks=callbacks, verbose=1
    )
    finally_time = time.time()

    print(f"Treinamento {version} concluído em {datetime.timedelta(seconds=finally_time - start_time)}")
    plot_training_history(history_base, history_fine, version)
    
    return model
