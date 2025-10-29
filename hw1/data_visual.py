import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def add_heatmap(in_data, name = 'image', title = ''):
    corr = in_data.corr()
    plt.figure(figsize=(15, 9))
    plt.title(f"{title}")
    sns_png = sns.heatmap(corr, annot=True, cmap='coolwarm')
    scatter_fig = sns_png.get_figure()
    scatter_fig.savefig(f'out_jpg/{name}.png')

def add_barplot(in_data, name = 'image', title = ''):
    plt.figure(figsize=(15, 9))
    plt.title(f"{title}")
    plt.tight_layout()
    sns_png = sns.barplot(data=in_data, x='Model', y='R2', palette='viridis')
    scatter_fig = sns_png.get_figure()
    scatter_fig.savefig(f'out_jpg/{name}.png')

def add_scatter(x_column, y_test, y_pred, name = 'image', title = ''):
    plt.figure(figsize=(15, 9))
    plt.scatter(x_column, y_test, color='blue', label='Фактические значения')
    plt.scatter(x_column, y_pred, color='red', label='Предсказанные значения')
    plt.grid(True)
    plt.xlabel('Наблюдение')
    plt.ylabel('Значение')
    plt.title(f"Фактические и предсказанные значения по регрессору {title}")
    plt.legend()
    plt.savefig(f'out_jpg/{name}.png')

def add_feature_importance(model, feature_names, name='feature_importance', title=''):

    if not hasattr(model, "feature_importances_"):
        print(f"Модель {model} не поддерживает feature_importances_")
        return

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=True) 

    plt.figure(figsize=(15, 9))
    sns_png = sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis')
    plt.title(title if title else f'Feature Importances')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    scatter_fig = sns_png.get_figure()
    scatter_fig.savefig(f'out_jpg/{name}.png')