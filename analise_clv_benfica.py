#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Análise CLV - SL Benfica
# Segmentação de sócios usando RFM e clustering

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Cores do SL Benfica
benfica_colors = ['#D30A0A', '#000000', '#FFFFFF', '#8B0000', '#CC0000']
sns.set_palette(benfica_colors)

# Carregar dados
df = pd.read_csv('dados_benfica.csv', sep=';', encoding='utf-8')
print(f"Dataset carregado: {len(df):,} sócios")

# Limpar nomes de colunas (remover espaços)
df.columns = df.columns.str.strip()
print(f"Colunas limpas: {len(df.columns)}")

# Funções de limpeza
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
    except:
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
    except:
        return 0

# Colunas RFM do Excel
rfm_cols_r = ['R_quotas_dias', 'R_merch_dias', 'R_bilh_dias', 'R_outros_dias', 'R_camp_dias', 'R_MercSec_dias']
rfm_cols_f = ['F_quotas', 'F_merch', 'F_bilh', 'F_outros', 'F_campan', 'F_MercSec']
rfm_cols_m = ['M_quotas', 'M_merch', 'M_bilh', 'M_outros', 'M_campn', 'M_MercSec']

# Verificar quais colunas existem
existing_r = []
for col in rfm_cols_r:
    if col in df.columns:
        existing_r.append(col)

existing_f = []
for col in rfm_cols_f:
    if col in df.columns:
        existing_f.append(col)
        
existing_m = []
for col in rfm_cols_m:
    if col in df.columns:
        existing_m.append(col)

# Limpar valores
for col in existing_r:
    df[col] = df[col].apply(clean_numeric)
    # Tratar 46049 como valor especial (nunca interagiu nessa categoria)
    # Mantemos o valor alto para indicar "nunca comprou"
    # Na normalização, estes terão score R=0 (pior recency possível)

for col in existing_f:
    df[col] = df[col].apply(clean_numeric)

for col in existing_m:
    df[col] = df[col].apply(clean_monetary)

# Normalização RFM (0-100)
def normalize_column(series, invert=False):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([50] * len(series))
    if invert:
        return ((max_val - series) / (max_val - min_val)) * 100
    else:
        return ((series - min_val) / (max_val - min_val)) * 100

for col in existing_r:
    new_col = col.replace('_dias', '_norm')
    df[new_col] = normalize_column(df[col], invert=True)

for col in existing_f:
    new_col = col + '_norm'
    df[new_col] = normalize_column(df[col], invert=False)

for col in existing_m:
    new_col = col + '_norm'
    df[new_col] = normalize_column(df[col], invert=False)

# Weighted RFM por categoria

weight_R = 0.248
weight_F = 0.343
weight_M = 0.409
categories = {
    'Quotas': {'R': 'R_quotas_norm', 'F': 'F_quotas_norm', 'M': 'M_quotas_norm'},
    'Merchandising': {'R': 'R_merch_norm', 'F': 'F_merch_norm', 'M': 'M_merch_norm'},
    'Bilhetica': {'R': 'R_bilh_norm', 'F': 'F_bilh_norm', 'M': 'M_bilh_norm'},
    'Outros': {'R': 'R_outros_norm', 'F': 'F_outros_norm', 'M': 'M_outros_norm'},
    'Campanhas': {'R': 'R_camp_norm', 'F': 'F_campan_norm', 'M': 'M_campn_norm'},
    'MercadoSec': {'R': 'R_MercSec_norm', 'F': 'F_MercSec_norm', 'M': 'M_MercSec_norm'}
}

# Calcular WRFM
for cat_name, cols in categories.items():
    r_col = cols['R']
    f_col = cols['F']
    m_col = cols['M']
    
    if r_col in df.columns and f_col in df.columns and m_col in df.columns:
        df[f'WRFM_{cat_name}'] = (
            df[r_col] * weight_R +
            df[f_col] * weight_F +
            df[m_col] * weight_M
        )

# RFM Global
print("\n5. CALCULANDO RFM GLOBAL...")

# Pesos por categoria
cat_weights = {
    'WRFM_Quotas': 0.30,
    'WRFM_Bilhetica': 0.25,
    'WRFM_Merchandising': 0.20,
    'WRFM_Outros': 0.10,
    'WRFM_Campanhas': 0.10,
    'WRFM_MercadoSec': 0.05
}

# Calcular RFM Global
wrfm_cols = []
for col in cat_weights.keys():
    if col in df.columns:
        wrfm_cols.append(col)
        
if len(wrfm_cols) > 0:
    # Média ponderada
    numerator = sum(df[col] * cat_weights.get(col, 1/len(wrfm_cols)) for col in wrfm_cols)
    denominator = sum(cat_weights.get(col, 1/len(wrfm_cols)) for col in wrfm_cols)
    df['RFM_Global'] = numerator / denominator
    print(f"   - RFM_Global calculado (usando {len(wrfm_cols)} categorias)")
else:
    # Fallback: média simples
    df['RFM_Global'] = 50
    print("   - RFM_Global = 50 (fallback)")

# CLV Histórico
print("\n6. CALCULANDO CLV...")

# Colunas de gastos monetários originais
monetary_cols_original = [
    ' € Gastos Quotas ', '€ Gastos Quotas',
    ' € Gastos Merchandising ', '€ Gastos Merchandising',
    ' € Gasto Bilhética Total ', '€ Gasto Bilhética Total',
    ' € Gastos Outros Serviços ', '€ Gastos Outros Serviços',
    ' € Campanhas ', '€ Campanhas',
    ' € Mercado Secundário ', '€ Mercado Secundário'
]

# Encontrar colunas que existem
existing_monetary = [col.strip() for col in df.columns if any(m.strip() in col for m in ['Gastos', 'Gasto', 'Campanhas', 'Mercado Secundário'])]
existing_monetary = [col for col in existing_monetary if '€' in col and 'Ticket' not in col and 'Redenção' not in col]

print(f"   Colunas monetárias encontradas: {existing_monetary}")

# Usar as colunas M que já limpámos
m_cols_clean = [col for col in existing_m if col in df.columns]

if len(m_cols_clean) > 0:
    df['CLV_Historico'] = df[m_cols_clean].sum(axis=1)
    print(f"   - CLV_Histórico calculado (soma de {len(m_cols_clean)} categorias)")
else:
    df['CLV_Historico'] = 0
    print("   - CLV_Histórico = 0 (nenhuma coluna M encontrada)")

# Estatísticas CLV
print(f"\n   Estatísticas CLV:")
print(f"      Média: {df['CLV_Historico'].mean():,.2f} €")
print(f"      Mediana: {df['CLV_Historico'].median():,.2f} €")
print(f"      Max: {df['CLV_Historico'].max():,.2f} €")
print(f"      Min: {df['CLV_Historico'].min():,.2f} €")

# Preparar dados para clustering
print("\n7. PREPARANDO CLUSTERING...")

# Features para clustering
cluster_features = ['RFM_Global', 'CLV_Historico']

# Adicionar WRFM por categoria se existirem
for col in ['WRFM_Quotas', 'WRFM_Merchandising', 'WRFM_Bilhetica', 'WRFM_Outros', 'WRFM_Campanhas', 'WRFM_MercadoSec']:
    if col in df.columns:
        cluster_features.append(col)

print(f"   Features para clustering: {cluster_features}")

# Criar matriz de features
X = df[cluster_features].fillna(0).values

# Standardizar (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   - Dados preparados: {X_scaled.shape[0]} sócios × {X_scaled.shape[1]} features")

# Determinar número ótimo de clusters
print("\n8. DETERMINANDO NÚMERO ÓTIMO DE CLUSTERS...")

K_range = range(2, 11)
wcss = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=123, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print(f"   K={k}: WCSS={kmeans.inertia_:,.0f}, Silhouette={sil_score:.3f}")

# Plot Elbow
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# WCSS (Elbow)
axes[0].plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (K)', fontsize=12)
axes[0].set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Método Elbow', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Silhouette
axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score por K', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_elbow_silhouette.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   - Gráfico salvo: 01_elbow_silhouette.png")

# Escolher K ótimo (maior silhouette ou K=8 como Chouaten)
best_k = K_range[np.argmax(silhouette_scores)]
print(f"\n   - K ótimo (maior silhouette): {best_k}")
print(f"   - K sugerido (baseado em Chouaten): 8")

# Usar K=8 para comparar com literatura, ou best_k
K_FINAL = 8  # Pode mudar para best_k
print(f"   - Usando K = {K_FINAL}")

# Clustering final
print(f"\n9. EXECUTANDO K-MEANS COM K={K_FINAL}...")

kmeans_final = KMeans(n_clusters=K_FINAL, random_state=123, n_init=20, max_iter=500)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Silhouette score final
sil_final = silhouette_score(X_scaled, df['Cluster'])
print(f"   - Clustering completo!")
print(f"   - Silhouette Score Final: {sil_final:.3f}")

# Contagem por cluster
print(f"\n   Distribuição por Cluster:")
for cluster in sorted(df['Cluster'].unique()):
    count = (df['Cluster'] == cluster).sum()
    pct = count / len(df) * 100
    print(f"      Cluster {cluster}: {count:,} sócios ({pct:.1f}%)")

# Caracterização dos clusters
print("\n10. CARACTERIZANDO CLUSTERS...")

# Métricas por cluster
cluster_summary = df.groupby('Cluster').agg({
    'CLV_Historico': ['mean', 'median', 'std', 'count'],
    'RFM_Global': ['mean', 'median'],
}).round(2)

# Flatten column names
new_cols = []
for col in cluster_summary.columns.values:
    new_cols.append('_'.join(col).strip())
cluster_summary.columns = new_cols

# Adicionar % de cada situação (se existir coluna Situação)
if 'Situação' in df.columns:
    for situacao in df['Situação'].unique():
        percentages = []
        for cluster in df['Cluster'].unique():
            cluster_data = df[df['Cluster'] == cluster]['Situação']
            count = (cluster_data == situacao).sum()
            pct = (count / len(cluster_data)) * 100
            percentages.append(pct)
        cluster_summary[f'%_{situacao}'] = percentages

print(cluster_summary)

# Salvar resumo
cluster_summary.to_excel('02_cluster_summary.xlsx')
print(f"\n   - Resumo salvo: 02_cluster_summary.xlsx")

# Nomear clusters
print("\n11. NOMEANDO CLUSTERS...")

# Ordenar clusters por CLV médio
cluster_clv_mean = df.groupby('Cluster')['CLV_Historico'].mean().sort_values(ascending=False)

# Nomes sugeridos (do maior CLV ao menor)
cluster_names_list = [
    'Champions',      # CLV muito alto
    'Golden Fans',    # CLV alto
    'Loyal',          # CLV médio-alto
    'Promising',      # CLV médio
    'Casual',         # CLV médio-baixo
    'New Fans',       # CLV baixo (podem ser novos)
    'At Risk',        # CLV baixo (podem ser inativos)
    'Dormant'         # CLV muito baixo
]

# Mapear clusters para nomes
cluster_name_map = {}
for i, (cluster_id, clv) in enumerate(cluster_clv_mean.items()):
    if i < len(cluster_names_list):
        cluster_name_map[cluster_id] = cluster_names_list[i]
    else:
        cluster_name_map[cluster_id] = f'Segment_{cluster_id}'

df['Cluster_Name'] = df['Cluster'].map(cluster_name_map)

print(f"   Mapeamento de clusters:")
sorted_clusters = sorted(cluster_name_map.items())
for cluster_id, name in sorted_clusters:
    clv = cluster_clv_mean[cluster_id]
    print(f"      Cluster {cluster_id} → {name} (CLV médio: {clv:,.2f} €)")

# Visualizações
print("\n12. CRIANDO VISUALIZAÇÕES...")

# 12.1 - Boxplot CLV por Cluster
plt.figure(figsize=(14, 6))
medians = df.groupby('Cluster_Name')['CLV_Historico'].median()
medians_sorted = medians.sort_values(ascending=False)
order = medians_sorted.index
sns.boxplot(data=df, x='Cluster_Name', y='CLV_Historico', order=order, palette='viridis')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('CLV Histórico (€)', fontsize=12)
plt.title('Distribuição de CLV por Cluster', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('03_boxplot_clv_cluster.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   - 03_boxplot_clv_cluster.png")

# 12.2 - Barplot CLV médio por Cluster
plt.figure(figsize=(12, 6))
cluster_means = df.groupby('Cluster_Name')['CLV_Historico'].mean().sort_values(ascending=False)
bars = plt.bar(cluster_means.index, cluster_means.values)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('CLV Médio (€)', fontsize=12)
plt.title('CLV Médio por Cluster', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Adicionar valores nas barras
for bar, val in zip(bars, cluster_means.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{val:,.0f}€', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('04_barplot_clv_medio.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   - 04_barplot_clv_medio.png")

# 12.3 - Heatmap características por cluster
plt.figure(figsize=(14, 8))
heatmap_cols = ['CLV_Historico', 'RFM_Global']
for col in cluster_features:
    if col not in heatmap_cols:
        heatmap_cols.append(col)

heatmap_data = df.groupby('Cluster_Name')[heatmap_cols].mean()
heatmap_data = heatmap_data.reindex(order)

# Normalizar para heatmap
heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

sns.heatmap(heatmap_normalized.T, annot=heatmap_data.T.round(1), fmt='', 
            cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Score Normalizado'})
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Métrica', fontsize=12)
plt.title('Características dos Clusters (Heatmap)', fontsize=14)
plt.tight_layout()
plt.savefig('05_heatmap_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   - 05_heatmap_clusters.png")

# 12.4 - Distribuição por Situação e Cluster (se existir)
if 'Situação' in df.columns:
    plt.figure(figsize=(14, 6))
    
    # Crosstab
    ct = pd.crosstab(df['Cluster_Name'], df['Situação'], normalize='index') * 100
    ct = ct.reindex(order)
    
    ct.plot(kind='bar', stacked=True, colormap='Set2', figsize=(14, 6))
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Percentagem (%)', fontsize=12)
    plt.title('Distribuição de Situação por Cluster', fontsize=14)
    plt.legend(title='Situação', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('06_situacao_por_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   - 06_situacao_por_cluster.png")

# 12.5 - Scatter RFM_Global vs CLV
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['RFM_Global'], df['CLV_Historico'], 
                      c=df['Cluster'], cmap='viridis', alpha=0.5, s=20)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('RFM Global Score', fontsize=12)
plt.ylabel('CLV Histórico (€)', fontsize=12)
plt.title('RFM Global vs CLV por Cluster', fontsize=14)
plt.tight_layout()
plt.savefig('07_scatter_rfm_clv.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   - 07_scatter_rfm_clv.png")

# 12.6 - Pie chart distribuição de sócios por cluster
plt.figure(figsize=(10, 10))
cluster_counts = df['Cluster_Name'].value_counts().reindex(order)
plt.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribuição de Sócios por Cluster', fontsize=14)
plt.tight_layout()
plt.savefig('08_pie_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   - 08_pie_clusters.png")

# Testes estatísticos
print("\n13. TESTES ESTATÍSTICOS...")

# Kruskal-Wallis para CLV entre clusters
clusters_clv = []
for i in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == i]['CLV_Historico'].values
    clusters_clv.append(cluster_data)
stat, p_value = stats.kruskal(*clusters_clv)
print(f"   Kruskal-Wallis (CLV entre clusters):")
print(f"      H-statistic: {stat:.2f}")
print(f"      p-value: {p_value:.2e}")
if p_value < 0.05:
    print(f"      - Diferenças SIGNIFICATIVAS entre clusters (p < 0.05)")
else:
    print(f"      - Diferenças NÃO significativas (p >= 0.05)")

# Exportar resultados
print("\n14. EXPORTANDO RESULTADOS...")

# Exportar dataset completo com clusters
df.to_excel('09_dataset_com_clusters.xlsx', index=False)
print(f"   - 09_dataset_com_clusters.xlsx")

# Exportar resumo detalhado
agg_dict = {
    'CLV_Historico': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'RFM_Global': ['mean', 'median', 'std']
}
if 'WRFM_Quotas' in df.columns:
    agg_dict['WRFM_Quotas'] = 'mean'
if 'WRFM_Merchandising' in df.columns:
    agg_dict['WRFM_Merchandising'] = 'mean'
if 'WRFM_Bilhetica' in df.columns:
    agg_dict['WRFM_Bilhetica'] = 'mean'
    
summary_detailed = df.groupby('Cluster_Name').agg(agg_dict).round(2)

summary_detailed.to_excel('10_cluster_detailed_summary.xlsx')
print(f"   - 10_cluster_detailed_summary.xlsx")

