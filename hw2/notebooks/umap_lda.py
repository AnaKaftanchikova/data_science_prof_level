import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import moduls.data_loader as data_loader
import moduls.log as log
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import seaborn as sns


def run():
    # 1. Загрузка
    logger = log.setup_logger()
    df = data_loader.load_data_csv("data/for_umap_lda/mushrooms.csv")
    logger.info(df.head())

    '''
    Информация об атрибутах: (классы: съедобные=e, ядовитые =p)
    - форма шляпки: колоколообразная =b, коническая = c, выпуклая = x, плоская = f, бугорчатая =k, утопленная =s
    - поверхность крышки: волокнистая=f, бороздки= g, чешуйчатая = y, гладкая= s
    - цвет шапочки: коричневый = n, темно-коричневый = b, коричный = c, серый = g, зеленый = r, розовый = p, фиолетовый = u, красный = e, белый = w, желтый = y
    - синяки: синяки =t, нет=f
    - запах: миндаль =a, анис =l, креозот =c, рыбный = y, неприятный =f, затхлый = m, отсутствует= n, острый = p, пряный =s
    - жабры: прикрепленные =a, нисходящие=d, свободные=f, зазубренные=n
    - расстояние между жабрами: близкое = c, переполненное = w, удаленное=d
    - размер жабр: широкие =b, узкие=n
    - цвет жабр: черный = k, коричневый = n, темно-коричневый = b, шоколадный = h, серый = g, зеленый = r, оранжевый = o, розовый = p, фиолетовый = u, красный = e, белый = w, желтый = y
    - форма стебля: расширяющийся =e, сужающийся=t
    - стебель-корень: луковичный = b, клубневидный = c, чашеобразный = u, равный = e, ризоморфный = z, укорененный=r, отсутствующий=?
    - поверхность стебля над кольцом: волокнистая=f, чешуйчатая=y, шелковистая=k, гладкая=s
    - поверхность стебля под кольцом: волокнистая=f, чешуйчатая=y, шелковистая=k,гладкая=s
    - цвет стебля над кольцом: коричневый = n, темно-коричневый = b, коричный = c, серый = g, оранжевый = o, розовый = p, красный = e, белый = w, желтый = y
    - цвет стебля под кольцом: коричневый=n, темно-коричневый=b, коричный =c, серый = g, оранжевый = o, розовый =p, красный =e, белый =w, желтый=y
    - тип вуали: частичная=p, универсальная=u
    - цвет вуали: коричневый=n, оранжевый = o, белый = w, желтый= y
    - номер кольца: нет=n, одно= o, два= t
    - тип кольца: паутинистое=c, затухающее=e, расширяющееся= f, большое = l, отсутствие= n, подвеска=p, оболочка=s, зона=z
    - цвет отпечатка спор: черный = k, коричневый = n, темно-коричневый = b, шоколадный = h, зеленый = r, оранжевый = o, фиолетовый = u, белый = w, желтый = y
    - население: многочисленное =a, сгруппированное =c, многочисленное = n, рассеянное = s, несколько= v, одиночное =y
    - среда обитания: травы = g, листья = l, луга = m, тропинки = p, города = u, пустоши = w, леса = d
    '''

    # 2. Предобработка
    label_encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    corr_matrix = df.corr()

    # 3. Визуализация матрицы корреляции
    plt.figure(figsize=(15,9))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Матрица корреляции", fontsize=16)
    plt.savefig(f'results/umap_lda/corr_matrix.png')

    features_to_drop = ["veil-type", "veil-color"]

    # Разделяем признаки и метки
    X = df.drop(features_to_drop + ["class"], axis=1)
    y = df["class"]

    # 4. Масштабирование признаков (для UMAP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X_scaled)

    # Визуализация UMAP
    plt.figure(figsize=(8,5))
    plt.scatter(X_umap[:,0], X_umap[:,1], c=y, cmap='tab10', alpha=0.7)
    plt.title("UMAP projection of Mushroom dataset")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(label='Class')
    plt.savefig(f'results/umap_lda/umap.png')

    # 6. LDA
    # LDA: n_components <= num_classes - 1
    lda_model = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda_model.fit_transform(X, y)

    # Визуализация LDA
    plt.figure(figsize=(8,5))
    plt.scatter(X_lda[:,0], np.zeros_like(X_lda[:,0]), c=y, cmap='tab10', alpha=0.7)
    plt.title("LDA projection of Mushroom dataset")
    plt.xlabel("LD1")
    plt.savefig(f'results/umap_lda/lda.png')

    # 7. Классификация (RandomForest)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logger.info("Classification Report:\n")
    logger.info(classification_report(y_test, y_pred))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f'results/umap_lda/confusion_matrix.png')