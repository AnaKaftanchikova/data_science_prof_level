import pandas as pd
import sqlite3
import os

def load_data_csv(file_path):
    return pd.read_csv(file_path)

def load_data_json(file_path):
    return pd.read_json(file_path)

def load_data_excel(file_path):
    return pd.read_excel(file_path)

def create_database(csv_path: str, db_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("bigmart_sales", conn, if_exists="replace", index=False)
    conn.close()
    print(f"База данных создана: {db_path}")
    return df

def load_data(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"База данных не найдена: {db_path}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM bigmart_sales", conn)
    conn.close()
    return df