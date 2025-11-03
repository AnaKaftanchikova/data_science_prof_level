import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    ["Данные", "Аналитика продаж", "Сравнение моделей"]
)

# === Главная страница ===
if page == "Данные":
    st.title(" BigMart Dataset Overview")
    df = load_data()
    st.write("### Первые строки данных", df.head())

    st.write("#### Общая информация:")
    st.write(f" Всего записей: {len(df)}")
    st.write(f" Столбцов: {len(df.columns)}")

    with st.expander("Показать статистику"):
        st.dataframe(df.describe())

# === Аналитика продаж ===
elif page == "Аналитика продаж":
    st.title(" Аналитика продаж")

    df = load_data()
    item_type = st.selectbox("Выберите категорию товара", df["Item_Type"].unique())
    filtered = df[df["Item_Type"] == item_type]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Средние продажи", f"{filtered['Item_Outlet_Sales'].mean():,.2f}")
    with col2:
        st.metric("Средняя цена", f"{filtered['Item_MRP'].mean():,.2f}")

    st.write("### Зависимость продаж от цены")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x="Item_MRP", y="Item_Outlet_Sales", data=filtered, alpha=0.7)
    plt.title(f"Зависимость продаж от цены для {item_type}")
    st.pyplot(fig)

# === Сравнение моделей ===
elif page == "Сравнение моделей":
    st.title(" Сравнение моделей машинного обучения")

    results_df = load_results()

    if results_df.empty:
        st.warning("Файл results.csv не найден. Сначала запустите main.py для обучения моделей.")
    else:
        st.write("### Таблица метрик")
        st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))

        st.write("### Визуальное сравнение моделей")
        metric = st.selectbox("Выберите метрику для сравнения:", ["R2", "RMSE", "MAE"])

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Model", y=metric, data=results_df.sort_values(metric, ascending=(metric != "R2")))
        plt.title(f"Сравнение моделей по {metric}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Лучшие модели
        best = results_df.sort_values("R2", ascending=False).iloc[0]
        st.success(f" Лучшая модель: **{best['Model']}** (R²={best['R2']:.3f})")
