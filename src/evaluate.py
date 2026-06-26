import os
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from config import RESULTS_DIR, CLINICAL_THRESHOLD, RELATORIOS_DIR, GRAFICOS_DIR

def evaluate_models_comparative(model_v1, test_gen_v1, model_v2, test_gen_v2, df_test):
    print("\nRealizando inferência no conjunto de teste (V1)...")
    test_gen_v1.reset()
    y_prob_v1 = model_v1.predict(test_gen_v1, verbose=1).reshape(-1)
    y_pred_v1 = (y_prob_v1 >= CLINICAL_THRESHOLD).astype(int)

    print("\nRealizando inferência no conjunto de teste (V2)...")
    test_gen_v2.reset()
    y_prob_v2 = model_v2.predict(test_gen_v2, verbose=1).reshape(-1)
    y_pred_v2 = (y_prob_v2 >= CLINICAL_THRESHOLD).astype(int)

    y_true = test_gen_v1.labels
    
    report_v1 = classification_report(y_true, y_pred_v1, target_names=["Saudável (0)", "Risco (1)"])
    report_v2 = classification_report(y_true, y_pred_v2, target_names=["Saudável (0)", "Risco (1)"])

    with open(os.path.join(RELATORIOS_DIR, "classification_report_comparativo.txt"), "w") as f:
        f.write("=== MODELO V1 ===\n")
        f.write(report_v1)
        f.write("\n\n=== MODELO V2 ===\n")
        f.write(report_v2)

    # 1b. Salvar também em formato CSV
    report_v1_dict = classification_report(y_true, y_pred_v1, target_names=["Saudável (0)", "Risco (1)"], output_dict=True)
    report_v2_dict = classification_report(y_true, y_pred_v2, target_names=["Saudável (0)", "Risco (1)"], output_dict=True)
    
    df_v1 = pd.DataFrame(report_v1_dict).transpose()
    df_v1['modelo'] = 'V1'
    df_v2 = pd.DataFrame(report_v2_dict).transpose()
    df_v2['modelo'] = 'V2'
    
    df_final = pd.concat([df_v1, df_v2])
    df_final.index.name = 'metric'
    df_final.reset_index(inplace=True)
    df_final = df_final[['modelo', 'metric', 'precision', 'recall', 'f1-score', 'support']]
    df_final.to_csv(os.path.join(RELATORIOS_DIR, "classification_report_comparativo.csv"), index=False)

    # 2. Plotagem Lado a Lado (Matrizes)
    fig_mat, axes_mat = plt.subplots(1, 2, figsize=(14, 5))

    # Matriz V1
    sns.heatmap(confusion_matrix(y_true, y_pred_v1), annot=True, fmt="d", cmap="Blues", ax=axes_mat[0])
    axes_mat[0].set_title("Matriz de Confusão - V1")
    axes_mat[0].set_ylabel("Real")
    axes_mat[0].set_xlabel("Predição")

    # Matriz V2
    sns.heatmap(confusion_matrix(y_true, y_pred_v2), annot=True, fmt="d", cmap="Blues", ax=axes_mat[1])
    axes_mat[1].set_title("Matriz de Confusão - V2")
    axes_mat[1].set_ylabel("Real")
    axes_mat[1].set_xlabel("Predição")

    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, "matriz_confusao_comparativa.png"))
    plt.close()

    # 3. Plotagem Lado a Lado (Curvas ROC)
    fig_roc, axes_roc = plt.subplots(1, 2, figsize=(14, 5))

    # ROC V1
    fpr_v1, tpr_v1, _ = roc_curve(y_true, y_prob_v1)
    axes_roc[0].plot(fpr_v1, tpr_v1, color="darkorange", lw=2, label=f"AUC = {auc(fpr_v1, tpr_v1):.3f}")
    axes_roc[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes_roc[0].set_title("Curva ROC - V1")
    axes_roc[0].legend(loc="lower right")

    # ROC V2
    fpr_v2, tpr_v2, _ = roc_curve(y_true, y_prob_v2)
    axes_roc[1].plot(fpr_v2, tpr_v2, color="darkorange", lw=2, label=f"AUC = {auc(fpr_v2, tpr_v2):.3f}")
    axes_roc[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes_roc[1].set_title("Curva ROC - V2")
    axes_roc[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, "curva_roc_comparativa.png"))
    plt.close()

    # 3. Exportação Unificada para CSV
    df_resultados = pd.DataFrame({
        'Filename': df_test['filename'],
        'Rotulo_Real': y_true,
        'Prob_V1': y_prob_v1,
        'Pred_V1': y_pred_v1,
        'Prob_V2': y_prob_v2,
        'Pred_V2': y_pred_v2
    })
    df_resultados.to_csv(os.path.join(RELATORIOS_DIR, "detalhamento_predicoes_v1_v2.csv"), index=False)

    print(f"Estatísticas comparativas salvas em: {RELATORIOS_DIR} e {GRAFICOS_DIR}")

