import pandas as pd
import numpy as np
import data_loader
import data_visual
import log
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostRegressor


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2, y_pred

def main():

    # Игнорирование предупреждений (опционально)
    warnings.filterwarnings('ignore')

    # dataset from https://www.kaggle.com/datasets/lovishbansal123/sales-of-a-supermarket
    file_path = "supermarket_sales.csv"

    #1
    log.add_log_info('=== ЗАГРУЗКА ДАННЫХ ===')

    try:
        data = data_loader.load_data_csv(file_path) 
        log.add_log_info('Чтение файла прошло успешно')
    except FileNotFoundError:
        log.add_log_error('Файл не найден')
    except Exception as e:
        log.add_log_error(f'Произошла ошибка: {e}')
    finally:
        log.add_log_debug('Загрузка данных закончена')

    is_null = pd.isnull(data)

    log.add_log_info(f"Размер датасета: {data.shape}")
    log.add_log_info(f"Первые строки:\n {data.head()}")
    # log.add_log_info(f"Типы данных: {data.info()}")
    log.add_log_info(f"Пропуски: {is_null.sum().sum()}\n")

    #2
    log.add_log_info('=== ПРЕДОБРАБОТКА ===')

    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday
    data['Hour'] = pd.to_datetime(data['Time']).dt.hour
    data['Is_weekend'] = (data['Weekday'] >= 5).astype(int)

    agg_product = data.groupby('Product line')['Total'].mean().rename('Product_avg_total')
    agg_customer = data.groupby('Customer type')['Total'].mean().rename('Customer_avg_total')
    agg_payment = data.groupby('Payment')['Total'].mean().rename('Payment_avg_total')

    data = data.merge(agg_product, on='Product line', how='left')
    data = data.merge(agg_customer, on='Customer type', how='left')
    data = data.merge(agg_payment, on='Payment', how='left')

    data['Product_Customer'] = data['Product line'].astype(str) + "_" + data['Customer type'].astype(str)

    data['Price_Quantity_log'] = np.log1p(data['Unit price'] * data['Quantity'])

    drop_cols = ['Invoice ID', 'Date', 'Time', 'cogs', 'Tax 5%', 'Unit price', 'Quantity',
                 'gross margin percentage', 'gross income', 'Total', 'Branch']

    log.add_log_debug('Разделение столбцов на числовые и категориальные')
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    log.add_log_debug('Обработка пропущенных значений для числовых столбцов')
    if data[numeric_columns].isnull().sum().any():
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    log.add_log_debug('Обработка пропущенных значений для категориальных столбцов')
    if data[categorical_columns].isnull().sum().any():
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    log.add_log_debug('Кодирование категориальных признаков с использованием LabelEncoder')
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    data_corr = data.drop(columns=drop_cols)

    try:
        numeric_data = data_corr.select_dtypes(include=[np.number])
        data_visual.add_heatmap(numeric_data, 'Corr_origin_info', 'Корреляция признаков с продажами')
        log.add_log_info('Матрица корреляции')
    except Exception as e:
        log.add_log_error(f'Произошла ошибка: {e}')
    finally:
        log.add_log_debug('finally_after_add_heatmap')


    X = data.drop(columns=drop_cols)
    y = data['Total']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    log.add_log_info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

    models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(alpha=0.1),
        "Ridge": Ridge(alpha=1.0),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    }

    results = []


    #3
    log.add_log_info("=== ОБУЧЕНИЕ МОДЕЛЕЙ ===")

    for name, model in models.items():
        # Обучаем на логарифмированной цели
        mae, rmse, r2, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        results.append((name, mae, rmse, r2))
        log.add_log_info(f"{name:20s} | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.3f}")

    log.add_log_info(f"  \n")

    #4
    log.add_log_info("=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
    results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
    log.add_log_info(results_df.sort_values(by="R2", ascending=False))

    
    #5
    log.add_log_info(f"  \n")
    log.add_log_info("=== ВИЗУАЛИЗАЦИЯ ===")

    try:
        data_visual.add_barplot(results_df, 'Bar_info', 'Сравнение моделей по R² (чем выше, тем лучше)')
        log.add_log_info('Столбчатая диаграмма')
    except Exception as e:
        log.add_log_error(f'Произошла ошибка: {e}')
    finally:
        log.add_log_debug('finally_after_add_barplot')


    best_model_name = results_df.sort_values(by="R2", ascending=False).iloc[0]["Model"]
    log.add_log_info(f"\nЛучшая модель: {best_model_name}")

    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred_best = np.expm1(best_model.predict(X_test))

    try:
        data_visual.add_scatter(range(len(y_test)), y_test, y_pred, 'Scatter_info', f"Прогноз продаж ({best_model_name})")
        log.add_log_info('Точечная диаграмма')
    except Exception as e:
        log.add_log_error(f'Произошла ошибка: {e}')
    finally:
        log.add_log_debug('finally_after_add_scatter')


    tree_models = ['DecisionTree', 'RandomForest', 'ExtraTrees', 'GradientBoosting', 'CatBoost']

    for model_name in tree_models:
        model = models[model_name]
        model.fit(X_train, y_train)  # обучаем на полном train
        data_visual.add_feature_importance(model, X_train.columns, 
                                        name=f'{model_name}_feature_importance', 
                                        title=f'{model_name} Feature Importance')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.add_log_error(f'Произошла ошибка: {e}')
    finally:
        log.add_log_debug('Done')
    