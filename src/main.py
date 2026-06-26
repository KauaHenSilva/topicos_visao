import os
from dataset import load_and_balance_data, get_data_generators
from train import train_pipeline
from evaluate import evaluate_models_comparative
from gradcam import generate_gradcam_comparative

def main():
    print("===================================================")
    print("INICIANDO PIPELINE DE CLASSIFICAÇÃO DE RETINA")
    print("===================================================\n")
    
    # 1. Carrega e prepara os dados
    df_train_balanceado, df_test, df_val, pesos_dit = load_and_balance_data()
    train_gen_v1, val_gen_v1, test_gen_v1 = get_data_generators(df_train_balanceado, df_val, df_test, version="V1")
    train_gen_v2, val_gen_v2, test_gen_v2 = get_data_generators(df_train_balanceado, df_val, df_test, version="V2")

    # 2. Treina os modelos
    model_v1 = train_pipeline(train_gen_v1, val_gen_v1, pesos_dit, version="V1")
    model_v2 = train_pipeline(train_gen_v2, val_gen_v2, pesos_dit, version="V2")

    # 3. Avalia os modelos e salva Classification Report, Matrizes e Curvas ROC
    evaluate_models_comparative(model_v1, test_gen_v1, model_v2, test_gen_v2, df_test)

    # 4. Gera Grad-CAM comparativo (agora totalmente encapsulado)
    generate_gradcam_comparative(df_test)

    # 5. Gera amostras categorizadas do Grad-CAM (Acertos e Erros)
    from gradcam import generate_categorized_gradcams
    generate_categorized_gradcams(df_test, num_samples=5)

    print("\n===================================================")
    print("PIPELINE CONCLUÍDA COM SUCESSO!")
    print("===================================================")

if __name__ == "__main__":
    main()
