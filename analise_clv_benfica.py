#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Análise CLV - SL Benfica
# Segmentação de sócios usando RFM e clustering

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# Formatador de eixos
# ---------------------------------------------------------------------------
def thousands_formatter(x, pos):
    """Formata números usando K para milhares."""
    if x >= 1000:
        return f'{x/1000:.0f}K'
    return f'{x:.0f}'


# ---------------------------------------------------------------------------
# Configurações de visualização
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

benfica_colors = ['#D30A0A', '#000000', '#FFFFFF', '#8B0000', '#CC0000']
sns.set_palette(benfica_colors)


# ---------------------------------------------------------------------------
# Funções de limpeza
# ---------------------------------------------------------------------------
def clean_monetary(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if value in ['-   €', '- €', '-  €', '-€', '-', '', 'nan']:
        return 0.0
    value = value.replace('€', '').replace(' ', '').replace(',', '.')
    try:
        return float(value)
    except Exception:
        return 0.0


def clean_numeric(value):
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if value in ['-', '', 'nan']:
        return 0
    try:
        return float(value)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Normalização RFM (0-100)
# ---------------------------------------------------------------------------
def normalize_column(series, invert=False):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([50] * len(series), index=series.index)
    if invert:
        return ((max_val - series) / (max_val - min_val)) * 100
    return ((series - min_val) / (max_val - min_val)) * 100


# ===========================================================================
# INÍCIO DA ANÁLISE
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. CARREGAR DADOS
# ---------------------------------------------------------------------------
print("\n1. CARREGANDO DADOS...")

DATA_FILE = 'dados_benfica.csv'
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Ficheiro '{DATA_FILE}' não encontrado.\n"
        f"Diretório atual: {os.getcwd()}\n"
        "Certifique-se de que o ficheiro está no mesmo diretório que este script."
    )

# utf-8-sig trata corretamente ficheiros CSV exportados do Excel em português (UTF-8 com BOM)
df = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8-sig')
df.columns = df.columns.str.strip()
print(f"   Dataset carregado: {len(df):,} sócios, {len(df.columns)} colunas")

# ---------------------------------------------------------------------------
# 2. IDENTIFICAR COLUNAS RFM
# ---------------------------------------------------------------------------
print("\n2. IDENTIFICANDO COLUNAS RFM...")

rfm_cols_r = ['R_quotas_dias', 'R_merch_dias', 'R_bilh_dias', 'R_outros_dias', 'R_camp_dias', 'R_MercSec_dias']
rfm_cols_f = ['F_quotas', 'F_merch', 'F_bilh', 'F_outros', 'F_campan', 'F_MercSec']
rfm_cols_m = ['M_quotas', 'M_merch', 'M_bilh', 'M_outros', 'M_campn', 'M_MercSec']

existing_r = [col for col in rfm_cols_r if col in df.columns]
existing_f = [col for col in rfm_cols_f if col in df.columns]
existing_m = [col for col in rfm_cols_m if col in df.columns]

print(f"   Colunas R encontradas: {existing_r}")
print(f"   Colunas F encontradas: {existing_f}")
print(f"   Colunas M encontradas: {existing_m}")

# ---------------------------------------------------------------------------
# 3. LIMPAR VALORES
# ---------------------------------------------------------------------------
print("\n3. LIMPANDO VALORES...")

# Valor sentinela do CRM da Benfica para sócios sem transações numa categoria.
# Na normalização invertida este valor resulta em R_norm ≈ 0 (pior recência possível).
MAX_RECENCY_SENTINEL = 46049

for col in existing_r:
    df[col] = df[col].apply(clean_numeric)

for col in existing_f:
    df[col] = df[col].apply(clean_numeric)

for col in existing_m:
    df[col] = df[col].apply(clean_monetary)

print("   Limpeza concluída.")

# ---------------------------------------------------------------------------
# 4. NORMALIZAÇÃO E WRFM POR CATEGORIA
# ---------------------------------------------------------------------------
print("\n4. NORMALIZANDO E CALCULANDO WRFM POR CATEGORIA...")

for col in existing_r:
    df[col.replace('_dias', '_norm')] = normalize_column(df[col], invert=True)

for col in existing_f:
    df[col + '_norm'] = normalize_column(df[col], invert=False)

for col in existing_m:
    df[col + '_norm'] = normalize_column(df[col], invert=False)

weight_R = 0.248
weight_F = 0.343
weight_M = 0.409

categories = {
    'Quotas':      {'R': 'R_quotas_norm', 'F': 'F_quotas_norm',  'M': 'M_quotas_norm'},
    'Merchandising':{'R': 'R_merch_norm',  'F': 'F_merch_norm',   'M': 'M_merch_norm'},
    'Bilhetica':   {'R': 'R_bilh_norm',   'F': 'F_bilh_norm',    'M': 'M_bilh_norm'},
    'Outros':      {'R': 'R_outros_norm', 'F': 'F_outros_norm',  'M': 'M_outros_norm'},
    'Campanhas':   {'R': 'R_camp_norm',   'F': 'F_campan_norm',  'M': 'M_campn_norm'},
    'MercadoSec':  {'R': 'R_MercSec_norm','F': 'F_MercSec_norm', 'M': 'M_MercSec_norm'},
}

for cat_name, cols in categories.items():
    r_col, f_col, m_col = cols['R'], cols['F'], cols['M']
    if r_col in df.columns and f_col in df.columns and m_col in df.columns:
        df[f'WRFM_{cat_name}'] = (
            df[r_col] * weight_R +
            df[f_col] * weight_F +
            df[m_col] * weight_M
        )
        print(f"   WRFM_{cat_name} calculado")

# ---------------------------------------------------------------------------
# 5. RFM GLOBAL
# ---------------------------------------------------------------------------
print("\n5. CALCULANDO RFM GLOBAL...")

cat_weights = {
    'WRFM_Quotas':       0.30,
    'WRFM_Bilhetica':    0.25,
    'WRFM_Merchandising':0.20,
    'WRFM_Outros':       0.10,
    'WRFM_Campanhas':    0.10,
    'WRFM_MercadoSec':   0.05,
}

wrfm_cols = [col for col in cat_weights if col in df.columns]

if wrfm_cols:
    numerator   = sum(df[col] * cat_weights[col] for col in wrfm_cols)
    denominator = sum(cat_weights[col] for col in wrfm_cols)
    df['RFM_Global'] = numerator / denominator
    print(f"   RFM_Global calculado (usando {len(wrfm_cols)} categorias)")
else:
    df['RFM_Global'] = 50
    print("   AVISO: RFM_Global = 50 (fallback — nenhuma coluna WRFM encontrada)")

# ---------------------------------------------------------------------------
# 6. CLV HISTÓRICO
# ---------------------------------------------------------------------------
print("\n6. CALCULANDO CLV HISTÓRICO...")

# CLV = soma das despesas monetárias nas 6 categorias (valor histórico acumulado)
m_cols_clean = [col for col in existing_m if col in df.columns]

if m_cols_clean:
    df['CLV_Historico'] = df[m_cols_clean].sum(axis=1)
    print(f"   CLV_Historico calculado (soma de {len(m_cols_clean)} categorias: {m_cols_clean})")
else:
    df['CLV_Historico'] = 0
    print("   AVISO: CLV_Historico = 0 (nenhuma coluna M encontrada)")

print(f"\n   Estatísticas CLV:")
print(f"      Média:   {df['CLV_Historico'].mean():,.2f} €")
print(f"      Mediana: {df['CLV_Historico'].median():,.2f} €")
print(f"      Máximo:  {df['CLV_Historico'].max():,.2f} €")
print(f"      Mínimo:  {df['CLV_Historico'].min():,.2f} €")

# ---------------------------------------------------------------------------
# 7. PREPARAR CLUSTERING
# ---------------------------------------------------------------------------
print("\n7. PREPARANDO CLUSTERING...")

cluster_features = ['RFM_Global', 'CLV_Historico']
for col in ['WRFM_Quotas', 'WRFM_Merchandising', 'WRFM_Bilhetica',
            'WRFM_Outros', 'WRFM_Campanhas', 'WRFM_MercadoSec']:
    if col in df.columns:
        cluster_features.append(col)

print(f"   Features para clustering ({len(cluster_features)}): {cluster_features}")

X = df[cluster_features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   Dados preparados: {X_scaled.shape[0]:,} sócios × {X_scaled.shape[1]} features")

# ---------------------------------------------------------------------------
# 8. NÚMERO ÓTIMO DE CLUSTERS
# ---------------------------------------------------------------------------
print("\n8. DETERMINANDO NÚMERO ÓTIMO DE CLUSTERS...")

# Fase de seleção: n_init=10 por eficiência (os valores WCSS da Tabela 3.3 da tese
# foram produzidos com esta configuração)
N_INIT_SEARCH = 10
K_range = range(2, 11)
wcss = []
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=123, n_init=N_INIT_SEARCH, max_iter=300)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    silhouette_scores.append(sil)
    print(f"   K={k}: WCSS={km.inertia_:,.0f}, Silhouette={sil:.3f}")

# Gráficos Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=16, fontweight='bold')
axes[0].set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=16, fontweight='bold')
axes[0].set_title('Elbow Method - WCSS vs Number of Clusters', fontsize=18, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=14)
axes[0].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=16, fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontsize=16, fontweight='bold')
axes[1].set_title('Silhouette Score by Number of Clusters', fontsize=18, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=14)

plt.tight_layout()
plt.savefig('01_elbow_silhouette.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Gráfico salvo: 01_elbow_silhouette.png")

best_k = list(K_range)[int(np.argmax(silhouette_scores))]
print(f"\n   K ótimo (maior silhouette): {best_k}")
print(f"   K selecionado (alinhado com Chouaten et al., 2024): 8")

# ---------------------------------------------------------------------------
# 9. K-MEANS FINAL (K=8)
# ---------------------------------------------------------------------------
K_FINAL = 8
print(f"\n9. EXECUTANDO K-MEANS FINAL COM K={K_FINAL}...")

# Modelo final: n_init=20 para maior robustez de convergência
N_INIT_FINAL = 20
kmeans_final = KMeans(n_clusters=K_FINAL, random_state=123, n_init=N_INIT_FINAL, max_iter=500)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

sil_final = silhouette_score(X_scaled, df['Cluster'])
print(f"   Silhouette Score Final: {sil_final:.3f}")

print(f"\n   Distribuição por Cluster:")
for cluster in sorted(df['Cluster'].unique()):
    count = (df['Cluster'] == cluster).sum()
    print(f"      Cluster {cluster}: {count:,} sócios ({count/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 10. CARACTERIZAÇÃO DOS CLUSTERS
# ---------------------------------------------------------------------------
print("\n10. CARACTERIZANDO CLUSTERS...")

cluster_summary = df.groupby('Cluster').agg({
    'CLV_Historico': ['mean', 'median', 'std', 'count'],
    'RFM_Global':    ['mean', 'median'],
}).round(2)
cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]

# Adicionar percentagens de status por cluster — com ordem garantida (sorted)
if 'Situação' in df.columns:
    for situacao in df['Situação'].unique():
        pct_dict = {}
        for cluster in sorted(df['Cluster'].unique()):   # ordem garantida: 0,1,...,7
            cluster_data = df[df['Cluster'] == cluster]['Situação']
            pct_dict[cluster] = (cluster_data == situacao).sum() / len(cluster_data) * 100
        cluster_summary[f'%_{situacao}'] = pd.Series(pct_dict)

print(cluster_summary)
cluster_summary.to_excel('02_cluster_summary.xlsx')
print("   Resumo salvo: 02_cluster_summary.xlsx")

# ---------------------------------------------------------------------------
# 11. NOMEAR CLUSTERS (por CLV médio decrescente)
# ---------------------------------------------------------------------------
print("\n11. NOMEANDO CLUSTERS...")

cluster_clv_mean = df.groupby('Cluster')['CLV_Historico'].mean().sort_values(ascending=False)

cluster_names_list = [
    'Champions',    # CLV muito alto
    'Golden Fans',  # CLV alto
    'Loyal',        # CLV médio-alto
    'Promising',    # CLV médio
    'Casual',       # CLV médio-baixo
    'New Fans',     # CLV baixo
    'At Risk',      # CLV baixo / inativo
    'Dormant',      # CLV muito baixo
]

cluster_name_map = {
    cluster_id: cluster_names_list[i] if i < len(cluster_names_list) else f'Segment_{cluster_id}'
    for i, (cluster_id, _) in enumerate(cluster_clv_mean.items())
}

df['Cluster_Name'] = df['Cluster'].map(cluster_name_map)

print("   Mapeamento de clusters (por CLV médio decrescente):")
for cluster_id, name in sorted(cluster_name_map.items()):
    clv = cluster_clv_mean[cluster_id]
    print(f"      Cluster {cluster_id} → {name} (CLV médio: {clv:,.2f} €)")

# Ordem de apresentação consistente (por CLV médio — igual ao critério de naming)
cluster_order = [cluster_name_map[cid] for cid, _ in cluster_clv_mean.items()]

# ---------------------------------------------------------------------------
# 12. VISUALIZAÇÕES
# ---------------------------------------------------------------------------
print("\n12. CRIANDO VISUALIZAÇÕES...")

# 12.1 — Boxplot CLV por Cluster (ordenado por CLV médio, consistente com naming)
plt.figure(figsize=(16, 10))
ax = sns.boxplot(data=df, x='Cluster_Name', y='CLV_Historico',
                 order=cluster_order, palette='viridis')
plt.xlabel('Cluster', fontsize=18, fontweight='bold')
plt.ylabel('Historical CLV (€)', fontsize=18, fontweight='bold')
plt.title('CLV Distribution by Cluster', fontsize=20, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=16)
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.tight_layout()
plt.savefig('03_boxplot_clv_cluster.png', dpi=300, bbox_inches='tight')
plt.close()
print("   03_boxplot_clv_cluster.png")

# 12.2 — Barplot CLV médio por Cluster
fig, ax = plt.subplots(figsize=(16, 10))
cluster_means = df.groupby('Cluster_Name')['CLV_Historico'].mean().reindex(cluster_order)
bars = plt.bar(cluster_means.index, cluster_means.values)
plt.xlabel('Cluster', fontsize=18, fontweight='bold')
plt.ylabel('Mean CLV (€)', fontsize=18, fontweight='bold')
plt.title('Mean CLV by Cluster', fontsize=20, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=16)
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
for bar, val in zip(bars, cluster_means.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
             f'{val:,.0f}€', ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('04_barplot_clv_medio.png', dpi=300, bbox_inches='tight')
plt.close()
print("   04_barplot_clv_medio.png")

# 12.3 — Heatmap características por cluster
plt.figure(figsize=(18, 12))
heatmap_cols = ['CLV_Historico', 'RFM_Global'] + [c for c in cluster_features if c not in ['CLV_Historico', 'RFM_Global']]
heatmap_data = df.groupby('Cluster_Name')[heatmap_cols].mean().reindex(cluster_order)
heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
sns.heatmap(heatmap_normalized.T, annot=heatmap_data.T.round(1), fmt='',
            cmap='YlOrRd', linewidths=0.5,
            cbar_kws={'label': 'Normalized Score'},
            annot_kws={'fontsize': 14})
plt.xlabel('Cluster', fontsize=18, fontweight='bold')
plt.ylabel('Feature', fontsize=18, fontweight='bold')
plt.title('Cluster Characteristics Heatmap', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('05_heatmap_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print("   05_heatmap_clusters.png")

# 12.4 — Membership Status Distribution by Cluster
if 'Situação' in df.columns:
    fig, ax = plt.subplots(figsize=(16, 10))
    ct = pd.crosstab(df['Cluster_Name'], df['Situação'], normalize='index') * 100
    ct = ct.reindex(cluster_order)
    ct = ct.rename(columns={'Quotas em Dia': 'Active', 'Quotas em Atraso': 'Overdue', 'Demitido': 'Churned'})
    ct.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)
    plt.xlabel('Cluster', fontsize=18, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=18, fontweight='bold')
    plt.title('Membership Status Distribution by Cluster', fontsize=20, fontweight='bold')
    plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('06_situacao_por_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   06_situacao_por_cluster.png")

# 12.5 — Scatter RFM Global vs CLV (com nomes de cluster na legenda)
fig, ax = plt.subplots(figsize=(16, 12))
color_map = {name: i for i, name in enumerate(cluster_order)}
color_values = df['Cluster_Name'].map(color_map)
scatter = plt.scatter(df['RFM_Global'], df['CLV_Historico'],
                      c=color_values, cmap='viridis', alpha=0.5, s=20)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_ticks(range(len(cluster_order)))
cbar.set_ticklabels(cluster_order)
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Cluster', fontsize=16, fontweight='bold')
plt.xlabel('RFM Global Score', fontsize=18, fontweight='bold')
plt.ylabel('Historical CLV (€)', fontsize=18, fontweight='bold')
plt.title('RFM Global Score vs. Historical CLV by Cluster', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.tight_layout()
plt.savefig('07_scatter_rfm_clv.png', dpi=300, bbox_inches='tight')
plt.close()
print("   07_scatter_rfm_clv.png")

# 12.6 — Pie chart: Member Distribution by Cluster
plt.figure(figsize=(14, 12))
cluster_counts = df['Cluster_Name'].value_counts().reindex(cluster_order)
plt.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 16, 'fontweight': 'bold'})
plt.title('Member Distribution by Cluster', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig('08_pie_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print("   08_pie_clusters.png")

# ---------------------------------------------------------------------------
# 13. TESTES ESTATÍSTICOS
# ---------------------------------------------------------------------------
print("\n13. TESTES ESTATÍSTICOS...")

clusters_clv = [df[df['Cluster'] == i]['CLV_Historico'].values
                for i in sorted(df['Cluster'].unique())]
stat, p_value = stats.kruskal(*clusters_clv)
print(f"   Kruskal-Wallis (CLV entre clusters):")
print(f"      H-statistic: {stat:.2f}")
print(f"      p-value:     {p_value:.2e}")
if p_value < 0.05:
    print("      -> Diferenças SIGNIFICATIVAS entre clusters (p < 0.05)")
else:
    print("      -> Diferenças NÃO significativas (p >= 0.05)")

# ---------------------------------------------------------------------------
# 14. EXPORTAR RESULTADOS
# ---------------------------------------------------------------------------
print("\n14. EXPORTANDO RESULTADOS...")

df.to_excel('09_dataset_com_clusters.xlsx', index=False)
print("   09_dataset_com_clusters.xlsx")

# Resumo detalhado — todas as 6 categorias WRFM incluídas
agg_dict = {
    'CLV_Historico': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'RFM_Global':    ['mean', 'median', 'std'],
}
all_wrfm = ['WRFM_Quotas', 'WRFM_Merchandising', 'WRFM_Bilhetica',
            'WRFM_Outros', 'WRFM_Campanhas', 'WRFM_MercadoSec']
for col in all_wrfm:
    if col in df.columns:
        agg_dict[col] = 'mean'

summary_detailed = df.groupby('Cluster_Name').agg(agg_dict).round(2)
summary_detailed.to_excel('10_cluster_detailed_summary.xlsx')
print("   10_cluster_detailed_summary.xlsx")

print("\n=== ANÁLISE CONCLUÍDA ===")
