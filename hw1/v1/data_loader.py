import pandas as pd

def load_data_csv(file_path):
    return pd.read_csv(file_path)

def load_data_json(file_path):
    return pd.read_json(file_path)

def load_data_excel(file_path):
    return pd.read_excel(file_path)