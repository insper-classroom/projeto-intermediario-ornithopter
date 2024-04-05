import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

DIRETORIO = 'results/'
PASTA_GRAFICOS = os.path.join(DIRETORIO, 'graficos')

# Cria a pasta para os gráficos se ela não existir
if not os.path.exists(PASTA_GRAFICOS):
    os.makedirs(PASTA_GRAFICOS)

def plot_graphs(df_combinado, titulo='Performance Comparison', nome_arquivo='Comparison.png'):
    sns.set(style="darkgrid")
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=df_combinado, x='episode', y='reward_mean', hue='Model', linewidth=2.5)
    plt.title(titulo, fontsize=16)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Mean of Rewards (Window Size = 50)', fontsize=14)
    plt.legend(title='Modelo', title_fontsize='13', fontsize='12', loc='upper left')
    ax.grid(True)
    
    caminho_arquivo = os.path.join(PASTA_GRAFICOS, nome_arquivo)
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()

# Listar todos os arquivos CSV no diretório especificado
arquivos = [arq for arq in os.listdir(DIRETORIO) if arq.endswith('.csv')]

# Agrupar arquivos pelo prefixo, assumindo que esteja relacionado ao experimento
names = {}
for arquivo in arquivos:
    prefixo = arquivo.split('_')[0]
    names.setdefault(prefixo, []).append(arquivo)

# Para cada grupo de arquivos relacionados a um experimento, combinar e plotar os gráficos
for prefixo, arquivos_grupo in names.items():
    dfs = []
    for arquivo in arquivos_grupo:
        df_temp = pd.read_csv(os.path.join(DIRETORIO, arquivo))
        df_temp['Model'] = arquivo.split('_')[-1].split('.')[0]
        dfs.append(df_temp)

    df_combinado = pd.concat(dfs, ignore_index=True)
    titulo_grafico = f'Performance Comparison: {prefixo}'
    nome_arquivo_saida = f'{prefixo}_Comparison.png'
    plot_graphs(df_combinado, titulo_grafico, nome_arquivo_saida)
