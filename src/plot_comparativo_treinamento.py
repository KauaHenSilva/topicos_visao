import os
import pandas as pd
import matplotlib.pyplot as plt

from config import GRAFICOS_DIR, RELATORIOS_DIR

def plot_comparative():
    csv_v1 = os.path.join(RELATORIOS_DIR, "historico_evolucao_v1.csv")
    csv_v2 = os.path.join(RELATORIOS_DIR, "historico_evolucao_v2.csv")
    
    if not os.path.exists(csv_v1) or not os.path.exists(csv_v2):
        print("Arquivos CSV de histórico não encontrados.")
        return

    df_v1 = pd.read_csv(csv_v1)
    df_v2 = pd.read_csv(csv_v2)

    epochs_v1 = range(1, len(df_v1) + 1)
    epochs_v2 = range(1, len(df_v2) + 1)

    # ===============================
    # GRÁFICO 1: APENAS TREINO
    # ===============================
    plt.figure(figsize=(16, 6))

    # Treino - Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_v1, df_v1['accuracy'], label='EfficientNetB0', color='#1f77b4', linewidth=2)
    plt.plot(epochs_v2, df_v2['accuracy'], label='EfficientNetB3', color='#ff7f0e', linewidth=2)
    plt.title('Treinamento: Acurácia (B0 vs B3)', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Treino - Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_v1, df_v1['loss'], label='EfficientNetB0', color='#1f77b4', linewidth=2)
    plt.plot(epochs_v2, df_v2['loss'], label='EfficientNetB3', color='#ff7f0e', linewidth=2)
    plt.title('Treinamento: Loss (B0 vs B3)', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    out_treino = os.path.join(GRAFICOS_DIR, "comparativo_apenas_treino.png")
    plt.savefig(out_treino)
    plt.close()

    # ===============================
    # GRÁFICO 2: APENAS VALIDAÇÃO
    # ===============================
    plt.figure(figsize=(16, 6))

    # Validação - Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_v1, df_v1['val_accuracy'], label='EfficientNetB0', color='#1f77b4', linewidth=2)
    plt.plot(epochs_v2, df_v2['val_accuracy'], label='EfficientNetB3', color='#ff7f0e', linewidth=2)
    plt.title('Validação: Acurácia (B0 vs B3)', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Validação - Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_v1, df_v1['val_loss'], label='EfficientNetB0', color='#1f77b4', linewidth=2)
    plt.plot(epochs_v2, df_v2['val_loss'], label='EfficientNetB3', color='#ff7f0e', linewidth=2)
    plt.title('Validação: Loss (B0 vs B3)', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    out_val = os.path.join(GRAFICOS_DIR, "comparativo_apenas_validacao.png")
    plt.savefig(out_val)
    plt.close()
    
    print(f"Gráficos gerados:\n- {out_treino}\n- {out_val}")

if __name__ == "__main__":
    plot_comparative()
