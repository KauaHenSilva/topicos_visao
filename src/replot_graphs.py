import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
from config import RESULTS_DIR, RELATORIOS_DIR, GRAFICOS_DIR

def main():
    csv_path = os.path.join(RELATORIOS_DIR, "detalhamento_predicoes_v1_v2.csv")
    if not os.path.exists(csv_path):
        print(f"Erro: O arquivo {csv_path} não foi encontrado.")
        return

    df = pd.read_csv(csv_path)

    y_true = df['Rotulo_Real']
    y_prob_v1 = df['Prob_V1']
    y_pred_v1 = df['Pred_V1']
    y_prob_v2 = df['Prob_V2']
    y_pred_v2 = df['Pred_V2']

    # Plotagem Lado a Lado (Matrizes)
    fig_mat, axes_mat = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(confusion_matrix(y_true, y_pred_v1), annot=True, fmt="d", cmap="Blues", ax=axes_mat[0])
    axes_mat[0].set_title("Matriz de Confusão - EfficientNetB0")
    axes_mat[0].set_ylabel("Real")
    axes_mat[0].set_xlabel("Predição")

    sns.heatmap(confusion_matrix(y_true, y_pred_v2), annot=True, fmt="d", cmap="Blues", ax=axes_mat[1])
    axes_mat[1].set_title("Matriz de Confusão - EfficientNetB3")
    axes_mat[1].set_ylabel("Real")
    axes_mat[1].set_xlabel("Predição")

    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, "matriz_confusao_comparativa.png"))
    plt.close()

    # Plotagem Lado a Lado (Curvas ROC)
    fig_roc, axes_roc = plt.subplots(1, 2, figsize=(14, 5))

    fpr_v1, tpr_v1, _ = roc_curve(y_true, y_prob_v1)
    axes_roc[0].plot(fpr_v1, tpr_v1, color="darkorange", lw=2, label=f"AUC = {auc(fpr_v1, tpr_v1):.3f}")
    axes_roc[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes_roc[0].set_title("Curva ROC - EfficientNetB0")
    axes_roc[0].legend(loc="lower right")

    fpr_v2, tpr_v2, _ = roc_curve(y_true, y_prob_v2)
    axes_roc[1].plot(fpr_v2, tpr_v2, color="darkorange", lw=2, label=f"AUC = {auc(fpr_v2, tpr_v2):.3f}")
    axes_roc[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes_roc[1].set_title("Curva ROC - EfficientNetB3")
    axes_roc[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, "curva_roc_comparativa.png"))
    plt.close()

    print("Gráficos separados gerados com sucesso a partir do CSV de predições!")

if __name__ == "__main__":
    main()
