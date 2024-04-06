import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

DIR = 'results/'
GRAPHS_FOLDER = os.path.join(DIR, 'graphs')

# Cria a pasta para os gráficos se ela não existir
if not os.path.exists(GRAPHS_FOLDER):
    os.makedirs(GRAPHS_FOLDER)

def plot_graphs(df_combined, title='Performance Comparison', file_name='comparison.png'):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=df_combined, x='episode', y='reward_mean', hue='model')
    plt.title(title, fontsize=16)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Mean of Rewards (Window Size = 50)', fontsize=14)
    plt.legend(title='Modelo', title_fontsize='13', fontsize='12', loc='upper left')
    ax.grid(True)
    
    path = os.path.join(GRAPHS_FOLDER, file_name)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# Listar todos os arquivos CSV no diretório especificado
csv_files = [arq for arq in os.listdir(DIR) if arq.endswith('.csv')]

# Agrupar arquivos pelo prefixo, assumindo que esteja relacionado ao experimento
names = {}
for file in csv_files:
    prefix = file.split('-')[0]
    names.setdefault(prefix, []).append(file)

# Para cada grupo de arquivos relacionados a um experimento, combinar e plotar os gráficos
for prefix, files_group in names.items():
    dfs = []
    for file in files_group:
        df_temp = pd.read_csv(os.path.join(DIR, file))
        dfs.append(df_temp)

    df_combined = pd.concat(dfs, ignore_index=True)
    graph_title = f'Performance Comparison: {prefix}'
    out_file_name = f'{prefix}_comparison.png'
    plot_graphs(df_combined, graph_title, out_file_name)
