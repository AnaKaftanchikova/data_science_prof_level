import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor

st.set_page_config(page_title="BigMart Analytics", layout="wide")

# === Функции ===
@st.cache_data
def load_data():
    conn = sqlite3.connect("db/bigmart.db")
    df = pd.read_sql("SELECT * FROM bigmart_sales", conn)
    conn.close()
    return df

@st.cache_data
def load_results():
    if os.path.exists("results.csv"):
        return pd.read_csv("results.csv")
    else:
        return pd.DataFrame()

# === Боковое меню ===
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел:",
    ["Данные", "Аналитика продаж", "Сравнение моделей", "EDA и Предсказание"]
)

# === Главная страница ===
if page == "Данные":
    st.title("BigMart Dataset Overview")
    df = load_data()
    st.write("### Первые строки данных", df.head())
    st.write("#### Общая информация:")
    st.write(f"Всего записей: {len(df)}")
    st.write(f"Столбцов: {len(df.columns)}")

    with st.expander("Показать статистику"):
        st.dataframe(df.describe())

# === Аналитика продаж ===
elif page == "Аналитика продаж":
    st.title("Аналитика продаж")
    df = load_data()
    item_type = st.selectbox("Выберите категорию товара", df["Item_Type"].unique())
    filtered = df[df["Item_Type"] == item_type]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Средние продажи", f"{filtered['Item_Outlet_Sales'].mean():,.2f}")
    with col2:
        st.metric("Средняя цена", f"{filtered['Item_MRP'].mean():,.2f}")

    st.write("### Зависимость продаж от цены")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x="Item_MRP", y="Item_Outlet_Sales", data=filtered, alpha=0.7)
    plt.title(f"Зависимость продаж от цены для {item_type}")
    st.pyplot(fig)

# === Сравнение моделей ===
elif page == "Сравнение моделей":
    st.title("Сравнение моделей машинного обучения")
    results_df = load_results()

    if results_df.empty:
        st.warning("Файл results.csv не найден. Сначала запустите main.py для обучения моделей.")
    else:
        st.write("### Таблица метрик")
        st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))

        st.write("### Визуальное сравнение моделей")
        metric = st.selectbox("Выберите метрику для сравнения:", ["R2", "RMSE", "MAE"])

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Model", y=metric, data=results_df.sort_values(metric, ascending=(metric != "R2")))
        plt.title(f"Сравнение моделей по {metric}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        best = results_df.sort_values("R2", ascending=False).iloc[0]
        st.success(f"Лучшая модель: **{best['Model']}** (R²={best['R2']:.3f})")

# === EDA и Предсказание ===
elif page == "EDA и Предсказание":
    # --- EDA и Предсказание ---
    st.subheader("EDA и Интерактивное предсказание продаж")
    df = load_data()

    # --- EDA ---
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = ["Item_Type", "Outlet_Type", "Outlet_Location_Type", "Outlet_Size"]

    # Гистограммы числовых признаков
    st.subheader("Гистограммы числовых признаков")
    for col in numeric_cols:
        with st.expander(f"Распределение {col}"):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

    # --- Предсказание ---
    st.subheader("Интерактивное предсказание продаж")

    features = ["Item_MRP", "Outlet_Establishment_Year", "Item_Type", "Outlet_Type", "Outlet_Location_Type", "Outlet_Size"]
    target = "Item_Outlet_Sales"

    X = df[features].copy()
    y = df[target]

    numeric_features = ["Item_MRP", "Outlet_Establishment_Year"]
    categorical_features = ["Item_Type", "Outlet_Type", "Outlet_Location_Type", "Outlet_Size"]

    # --- Заполнение пропусков и приведение типов ---
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())
    X[categorical_features] = X[categorical_features].fillna("Unknown").astype(str)

    # --- Ввод признаков с заполнением по умолчанию ---
    st.write("Введите параметры товара:")

    # Числовые признаки
    item_mrp = st.number_input(
        "Цена товара (Item_MRP)",
        min_value=float(df["Item_MRP"].min()),
        max_value=float(df["Item_MRP"].max()),
        value=float(df["Item_MRP"].mean())
    )

    outlet_year = st.number_input(
        "Год открытия магазина (Outlet_Establishment_Year)",
        min_value=int(df["Outlet_Establishment_Year"].min()),
        max_value=int(df["Outlet_Establishment_Year"].max()),
        value=int(df["Outlet_Establishment_Year"].mean())
    )

    # Категориальные признаки
    def selectbox_safe(label, df_col):
        options = df_col.dropna().unique().tolist()
        default_val = df_col.mode()[0] if not df_col.mode().empty else options[0]
        return st.selectbox(label, options=options, index=options.index(default_val))

    item_type_pred = selectbox_safe("Категория товара (Item_Type)", df["Item_Type"])
    outlet_type_pred = selectbox_safe("Тип магазина (Outlet_Type)", df["Outlet_Type"])
    outlet_loc_pred = selectbox_safe("Локация магазина (Outlet_Location_Type)", df["Outlet_Location_Type"])
    outlet_size_pred = selectbox_safe("Размер магазина (Outlet_Size)", df["Outlet_Size"])

    # Количество единиц товара
    item_quantity = st.number_input("Количество товаров для прогноза", min_value=1, value=1)

    # Создаем DataFrame для предсказания
    pred_df = pd.DataFrame({
        "Item_MRP": [item_mrp],
        "Outlet_Establishment_Year": [outlet_year],
        "Item_Type": [item_type_pred],
        "Outlet_Type": [outlet_type_pred],
        "Outlet_Location_Type": [outlet_loc_pred],
        "Outlet_Size": [outlet_size_pred]
    })

    pred_df[categorical_features] = pred_df[categorical_features].astype(str)

    # Выбираем модель
    model_type = st.radio("Выберите модель:", ["Линейная регрессия", "CatBoost"])

    if model_type == "Линейная регрессия":
        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
            remainder="passthrough"
        )
        model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
        model.fit(X, y)
    else:
        model = CatBoostRegressor(verbose=0, random_state=42)
        model.fit(X, y, cat_features=categorical_features)

    # Предсказание
    prediction = model.predict(pred_df)[0]

    # Привязка к количеству товаров
    total_sales = prediction * item_quantity
    per_item_sales = prediction

    st.success(f"Предсказанные продажи за {item_quantity} единиц: {total_sales:,.2f}")
    st.info(f"Примерно на 1 товар: {per_item_sales:,.2f}")
