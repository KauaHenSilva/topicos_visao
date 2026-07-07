import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from config import TRAIN_CSV, VAL_CSV, TEST_CSV, GRAFICOS_DIR

def generate_stats_table_image():
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    total_images = len(df_all)
    
    disease_cols = [c for c in df_all.columns if c not in ['ID', 'Disease_Risk']]
    
    stats = []
    for col in disease_cols:
        count = df_all[col].sum()
        pct = (count / total_images) * 100
        stats.append((col, count, pct))
        
    stats.sort(key=lambda x: x[1], reverse=True)
    
    # Mapeamento para nomes conhecidos
    mapping = {
        'DR': 'Diabetic Retinopathy',
        'MH': 'Macular Hole',
        'ODC': 'Optic Disc Cupping',
        'TSLN': 'Tessellation',
        'DN': 'Drusen',
        'ARMD': 'Age-related Macular Degeneration',
        'MYA': 'Myopia',
        'BRVO': 'Branch Retinal Vein Occlusion',
        'ODP': 'Optic Disc Pallor',
        'ODE': 'Optic Disc Edema',
        'LS': 'Laser Scars',
        'RS': 'Retinitis',
        'CSR': 'Central Serous Retinopathy',
        'CRS': 'Chorioretinitis',
        'CRVO': 'Central Retinal Vein Occlusion',
        'RPEC': 'RPE Changes',
        'MS': 'Macular Scar',
        'ERM': 'Epiretinal Membrane',
        'AION': 'Anterior Ischemic Optic Neuropathy',
        'AH': 'Asteroid Hyalosis',
        'RT': 'Retinal Tear',
        'EDN': 'Exudates',
        'PT': 'Phthisis Bulbi',
        'MHL': 'Macular Hole Localized',
        'ST': 'Staphyloma',
        'TV': 'Tortuous Vessels',
        'RP': 'Retinitis Pigmentosa',
        'TD': 'Tractional Detachment',
        'CWS': 'Cotton Wool Spots',
        'CME': 'Cystoid Macular Edema',
        'PTCR': 'Post-Traumatic Choroidal Rupture',
        'CF': 'Choroidal Folds',
        'PRH': 'Preretinal Hemorrhage',
        'CRAO': 'Central Retinal Artery Occlusion',
        'VH': 'Vitreous Hemorrhage',
        'VS': 'Vitreous Syneresis',
        'BRAO': 'Branch Retinal Artery Occlusion',
        'MNF': 'Myelinated Nerve Fibers',
        'CB': 'Coloboma',
        'ODPM': 'Optic Disc Pit Maculopathy',
        'PLQ': 'Plaque',
        'CL': 'Choroidal Lesion',
        'HR': 'Hypertensive Retinopathy',
        'MCA': 'Macroaneurysm',
        'HPED': 'Hemorrhagic Pigment Epithelial Detachment'
    }
    
    # Formata as linhas aplicando wrap aos nomes
    formatted_rows = []
    for col, count, pct in stats:
        desc = mapping.get(col, "-")
        desc_wrapped = "\n".join(textwrap.wrap(desc, width=30))
        formatted_rows.append([col, desc_wrapped, str(count), f"{pct:.2f}%"])

    # Divide em 3 blocos lado a lado para ficar "lateralizada" e não "extensa para baixo"
    num_blocks = 3
    rows_per_block = 15
    
    while len(formatted_rows) < num_blocks * rows_per_block:
        formatted_rows.append(["", "", "", ""])

    table_data = []
    # Cabeçalho unificado para os 3 blocos com colunas de espaçamento invisíveis
    header = []
    for b in range(num_blocks):
        header.extend(["Sigla", "Nome", "Qtd", "%"])
        if b < num_blocks - 1:
            header.append("") # Coluna vazia (espaçador)
            
    table_data.append(header)

    # Preenche as linhas buscando os itens sequencialmente nos 3 blocos
    for i in range(rows_per_block):
        row = []
        for b in range(num_blocks):
            idx = b * rows_per_block + i
            row.extend(formatted_rows[idx])
            if b < num_blocks - 1:
                row.append("") # Coluna vazia
        table_data.append(row)

    # Criação da figura no Matplotlib em formato "paisagem" / horizontal
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # As larguras proporcionais (diminuí o spacer e aumentei o 'Nome')
    col_widths = []
    for b in range(num_blocks):
        col_widths.extend([0.04, 0.22, 0.04, 0.05])
        if b < num_blocks - 1:
            col_widths.append(0.015) # Largura do espaçamento reduzida
    
    table = ax.table(cellText=table_data, 
                     colLabels=None, 
                     loc='center', 
                     cellLoc='center',
                     colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    # Aumenta a altura para padronizar e acomodar as quebras de linha
    table.scale(1, 3.0)
    
    # Formatação padronizada e bonita
    for i in range(len(table_data)):
        for j in range(len(header)):
            cell = table[(i, j)]
            
            # Se for uma coluna de espaçamento (índices 4 e 9 num layout de 14 colunas), deixa invisível
            if j == 4 or j == 9:
                cell.set_facecolor('white')
                cell.set_edgecolor('white')
                continue

            if i == 0:
                # Cor do cabeçalho
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4A4A4A')
                cell.set_edgecolor('#4A4A4A')
            else:
                # Cores intercaladas (Zebra stripes) para facilitar leitura
                if i % 2 == 0:
                    cell.set_facecolor('#F3F3F3')
                else:
                    cell.set_facecolor('#FFFFFF')
                
                # Borda sutil
                cell.set_edgecolor('#DDDDDD')

    plt.tight_layout()
    out_path = os.path.join(GRAFICOS_DIR, "tabela_todas_doencas.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Tabela gerada em: {out_path}")

if __name__ == "__main__":
    generate_stats_table_image()
