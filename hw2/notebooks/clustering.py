import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import moduls.data_loader as data_loader
import moduls.log as log
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder


def run():
    # 1. Загрузка
    logger = log.setup_logger()
    df = data_loader.load_data_csv("data/for_clustering/Mall_Customers.csv")
    logger.info(df.head())

    # Кодируем Gender
    # df['Gender_Code'] = LabelEncoder().fit_transform(df['Genre'])

    # 2. Предобработка
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] #'Gender_Code', 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Иерархическая кластеризация
    linked = linkage(X_scaled, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.axhline(y=7.5, color='r', linestyle='--', label='Уровень среза (y=10)')
    plt.title("Дендрограмма (иерархическая кластеризация)")
    plt.xlabel("Наблюдения")
    plt.ylabel("Евклидово расстояние")
    plt.savefig(f'results/clustering/dendrogram.png')

    # Оптимальное количество кластеров (например, по дендрограмме — 5)
    agg = AgglomerativeClustering(n_clusters=6)
    labels_agg = agg.fit_predict(X_scaled)
    df['Cluster_Agg'] = labels_agg

    for eps in [0.3, 0.5, 0.8, 1.0, 1.2]:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X_scaled)
        logger.info(f"eps={eps} → кластеры: {set(labels)}, counts:", np.bincount(labels[labels >= 0]))

    # 4. DBSCAN
    db = DBSCAN(eps=0.5, min_samples=6)
    labels_db = db.fit_predict(X_scaled)
    df['Cluster_DBSCAN'] = labels_db

    # 5. Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(ax=axes[0],
                    x='Annual Income (k$)', y='Spending Score (1-100)',
                    hue='Cluster_Agg', palette='Set2', data=df)
    axes[0].set_title("Agglomerative Clustering")
    sns.scatterplot(ax=axes[1],
                    x='Annual Income (k$)', y='Spending Score (1-100)',
                    hue='Cluster_DBSCAN', palette='Set2', data=df)
    axes[1].set_title("DBSCAN Clustering")
    plt.savefig(f'results/clustering/dbscan_vs_agg_clust.png')

    # 6. Сравнение результатов
    logger.info("Agglomerative кластеризация:")
    logger.info(df['Cluster_Agg'].value_counts())
    logger.info("\nDBSCAN:")
    logger.info(df['Cluster_DBSCAN'].value_counts())