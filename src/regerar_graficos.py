import os
import pandas as pd
import matplotlib.pyplot as plt

from config import GRAFICOS_DIR, RELATORIOS_DIR, EPOCAS_BASE

def regenerate_plot(version):
    csv_path = os.path.join(RELATORIOS_DIR, f"historico_evolucao_{version.lower()}.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    acc = df['accuracy'].values
    val_acc = df['val_accuracy'].values
    loss = df['loss'].values
    val_loss = df['val_loss'].values

    # A transição real do V1 foi na época 6 e V2 na época 7.
    if version == "V1":
        transicao_idx = 6
        model_name = "EfficientNetB0"
    else:
        transicao_idx = 7
        model_name = "EfficientNetB3"

    epochs_base = list(range(1, transicao_idx + 1))
    epochs_fine = list(range(1, len(acc) - transicao_idx + 1)) # Começa do 1 para o finetune também para ficar bonito, ou mantém a época global? O user prefere separado, entao vamos usar épocas 1..N

    # ==========================
    # GRÁFICO DA FASE BASE
    # ==========================
    plt.figure(figsize=(14, 6))

    # --- Gráfico de Acurácia (Base) ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_base, acc[:transicao_idx], label='Treino', linewidth=2, color='#1f77b4')
    plt.plot(epochs_base, val_acc[:transicao_idx], label='Validação', linewidth=2, color='#ff7f0e')
    plt.title(f'Evolução da Acurácia - {model_name} (Base)', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # --- Gráfico de Loss (Base) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_base, loss[:transicao_idx], label='Treino', linewidth=2, color='#1f77b4')
    plt.plot(epochs_base, val_loss[:transicao_idx], label='Validação', linewidth=2, color='#ff7f0e')
    plt.title(f'Evolução do Loss - {model_name} (Base)', fontsize=14)
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
        plt.title(f'Evolução da Acurácia - {model_name} (Fine-Tune)', fontsize=14)
        plt.xlabel('Épocas', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

        # --- Gráfico de Loss (Fine-Tune) ---
        plt.subplot(1, 2, 2)
        plt.plot(epochs_fine, loss[transicao_idx:], label='Treino', linewidth=2, color='#1f77b4')
        plt.plot(epochs_fine, val_loss[transicao_idx:], label='Validação', linewidth=2, color='#ff7f0e')
        plt.title(f'Evolução do Loss - {model_name} (Fine-Tune)', fontsize=14)
        plt.xlabel('Épocas', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(GRAFICOS_DIR, f"evolucao_treinamento_{version.lower()}_finetune.png"))
        plt.close()
    
    print(f"Regenerated plots for: {version}")

if __name__ == "__main__":
    regenerate_plot("V1")
    regenerate_plot("V2")
