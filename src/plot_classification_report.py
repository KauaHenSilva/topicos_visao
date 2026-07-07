import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_report_table_image():
    csv_path = r"C:\_Projetos\Pessoal\topicos\notebooks\resultados_modelo\relatorios\classification_report_comparativo.csv"
    out_path = r"C:\_Projetos\Pessoal\topicos\notebooks\resultados_modelo\graficos\tabela_classification_report.png"
    
    if not os.path.exists(csv_path):
        print(f"Arquivo não encontrado: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    table_data = []
    # Novo Cabeçalho conforme solicitado
    table_data.append([
        "Modelo", 
        "Classe / Métrica", 
        "Precision\n(Precisão)", 
        "Recall\n(Sensibilidade)", 
        "F1-Score", 
        "Support\n(Amostras)"
    ])
    
    # Mapeamento do modelo com os nomes reais (você havia pedido pra trocar V1 e V2 antes, então vou manter os nomes reais, mas a estrutura que você mandou)
    for model_name, new_name in zip(['V1', 'V2'], ['EfficientNetB0', 'EfficientNetB3']):
        df_model = df[df['modelo'] == model_name]
        
        # 1. Saudável (0)
        row_0 = df_model[df_model['metric'] == 'Saudável (0)'].iloc[0]
        table_data.append([
            new_name, 
            "Saudável (0)", 
            f"{row_0['precision']:.2f}", 
            f"{row_0['recall']:.2f}", 
            f"{row_0['f1-score']:.2f}", 
            str(int(row_0['support']))
        ])
        
        # 2. Risco (1)
        row_1 = df_model[df_model['metric'] == 'Risco (1)'].iloc[0]
        table_data.append([
            "", 
            "Risco (1)", 
            f"{row_1['precision']:.2f}", 
            f"{row_1['recall']:.2f}", 
            f"{row_1['f1-score']:.2f}", 
            str(int(row_1['support']))
        ])
        
        # 3. Macro Avg
        row_macro = df_model[df_model['metric'] == 'macro avg'].iloc[0]
        table_data.append([
            "", 
            "Média Macro", 
            f"{row_macro['precision']:.2f}", 
            f"{row_macro['recall']:.2f}", 
            f"{row_macro['f1-score']:.2f}", 
            "640"
        ])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data, 
                     loc='center', 
                     cellLoc='center',
                     colWidths=[0.22, 0.22, 0.16, 0.18, 0.12, 0.16])
                     
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Formatação
    for i in range(len(table_data)):
        for j in range(6):
            cell = table[(i, j)]
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2C3E50')
                cell.set_edgecolor('#2C3E50')
            else:
                # Pintar fundo suavemente: azul para B0 (linhas 1,2,3), vermelho para B3 (linhas 4,5,6)
                if i in [1, 2, 3]:
                    cell.set_facecolor('#F0F8FF') # Alice Blue
                else:
                    cell.set_facecolor('#FFF0F5') # Lavender Blush
                    
                cell.set_edgecolor('#D5D8DC')
                
                # Alinhar as colunas de Modelo e Classe à esquerda
                if j in [0, 1]:
                    cell._loc = 'center' if i == 0 else 'left'
                    
                # Negrito para a Acurácia Geral e Macro Avg
                if table_data[i][1] in ["Acurácia Geral", "Média Macro"]:
                    cell.set_text_props(weight='bold')
                
                # Negrito para o nome do modelo
                if j == 0 and table_data[i][0] != "":
                    cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Tabela salva em: {out_path}")

if __name__ == "__main__":
    generate_report_table_image()
