import pandas as pd
import numpy as np
import warnings
import moduls.data_loader as data_loader
import moduls.data_visual as data_visual
import moduls.log as log

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor

import optuna

warnings.filterwarnings("ignore")

def main():
    logger = log.setup_logger()
    logger.info("=== Запуск программы ===")

    try:
        # 1. Загрузка данных
        df = data_loader.create_database(
            csv_path="data/bigmart_sales_clean.csv",
            db_path="db/bigmart.db"
        )
        logger.info(f"1. Данные загружены: {df.shape}")

        df = df.dropna(how='all')

        # Заполнение пропусков
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")

        # Приведение категорий
        if 'Item_Fat_Content' in df.columns:
            df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
                'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
            })

        # === Генерация признаков ===
        if 'Outlet_Establishment_Year' in df.columns:
            df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
            df['Outlet_Age_Squared'] = df['Outlet_Age'] ** 2
            df['Outlet_Age_Log'] = np.log1p(df['Outlet_Age'])

        if 'Item_Identifier' in df.columns:
            df["Item_Category"] = df["Item_Identifier"].str[:2].map({
                'FD': 'Food', 'DR': 'Drinks', 'NC': 'Non-Consumable'
            }).fillna('Other')

        if 'Item_MRP' in df.columns:
            df['Log_Item_MRP'] = np.log1p(df['Item_MRP'])
            df['MRP_Squared'] = df['Item_MRP'] ** 2
            df['MRP_per_Weight'] = df['Item_MRP'] / (df['Item_Weight'] + 1e-5)

        if 'Item_Visibility' in df.columns:
            df['Item_Visibility'] = df['Item_Visibility'].replace(0, np.nan)
            df['Sqrt_Visibility'] = np.sqrt(df['Item_Visibility'])
            df['Log_Visibility'] = np.log1p(df['Item_Visibility'])
            df['Item_Visibility'] = df['Item_Visibility'].clip(upper=df['Item_Visibility'].quantile(0.99))
            df['Visibility_per_Weight'] = df['Item_Visibility'] / (df['Item_Weight'] + 1e-5)

        # Категориальные взаимодействия
        df['Outlet_Location_Type_Item_Type'] = df['Outlet_Location_Type'].astype(str) + "_" + df['Item_Type'].astype(str)
        df['Outlet_Type_Item_Fat_Content'] = df['Outlet_Type'].astype(str) + "_" + df['Item_Fat_Content'].astype(str)
        df['Outlet_Item_Combo'] = df['Outlet_Type'].astype(str) + "_" + df['Item_Category'].astype(str)

        # Разделение на X и y до создания target-like признаков
        # Разделение данных (увеличиваем обучающую выборку до 90%) с стратификацией по Item_Type
        X = df.drop("Item_Outlet_Sales", axis=1)
        y = df["Item_Outlet_Sales"]

        # Стратификация требует категориального признака, поэтому используем исходный Item_Type
        if 'Item_Type' in df.columns:
            stratify_col = df['Item_Type']
        else:
            stratify_col = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=stratify_col
        )
        logger.info(" 2. Предобработка окончена, train_size увеличен до 90%, стратификация по Item_Type выполнена")


        # === Безопасные агрегированные признаки ===
        for col in ['Item_Type', 'Outlet_Type', 'Item_Category', 'Outlet_Identifier']:
            # среднее по тренировочному набору
            means = X_train.join(y_train).groupby(col)["Item_Outlet_Sales"].mean()
            X_train[f'{col}_Sales_Mean'] = X_train[col].map(means)
            X_test[f'{col}_Sales_Mean'] = X_test[col].map(means)

        # Медиана / std по магазинам
        outlet_stats = X_train.join(y_train).groupby("Outlet_Identifier")["Item_Outlet_Sales"].agg(["median", "std", "count"])
        X_train["Outlet_Sales_Median"] = X_train["Outlet_Identifier"].map(outlet_stats["median"])
        X_train["Outlet_Sales_Std"] = X_train["Outlet_Identifier"].map(outlet_stats["std"])
        X_train["Outlet_Item_Count"] = X_train["Outlet_Identifier"].map(outlet_stats["count"])
        X_test["Outlet_Sales_Median"] = X_test["Outlet_Identifier"].map(outlet_stats["median"])
        X_test["Outlet_Sales_Std"] = X_test["Outlet_Identifier"].map(outlet_stats["std"])
        X_test["Outlet_Item_Count"] = X_test["Outlet_Identifier"].map(outlet_stats["count"])

        # Label Encoding
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le

        numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train[numeric_columns] = X_train[numeric_columns].fillna(X_train[numeric_columns].median())
        X_test[numeric_columns] = X_test[numeric_columns].fillna(X_train[numeric_columns].median())

        logger.info("2. Предобработка окончена")

        # === Модели ===
        models = {
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso(alpha=0.1),
            "Ridge": Ridge(alpha=1.0),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=400, max_depth=12, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, random_state=42),
        }

        # CatBoost с Optuna
        def objective_catboost(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 800, 1500),
                "depth": trial.suggest_int("depth", 6, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_strength": trial.suggest_float("random_strength", 0.5, 1.5),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 1.0),
                "verbose": 0,
                "random_state": 42
            }
            model = CatBoostRegressor(**params)
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, np.log1p(y_train), cv=cv, scoring="r2")
            return np.mean(scores)

        study_cb = optuna.create_study(direction="maximize")
        study_cb.optimize(objective_catboost, n_trials=30)
        best_cb = CatBoostRegressor(**study_cb.best_params, verbose=0, random_state=42)
        best_cb.fit(X_train, np.log1p(y_train))
        models["CatBoost_Opt"] = best_cb

        # === Обучение и метрики ===
        results = []
        for name, model in models.items():
            logger.info(f"Обучение модели: {name}")
            y_train_log = np.log1p(y_train)
            y_test_log = np.log1p(y_test)
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
            logger.info(f"{name} → R2={r2:.3f}, RMSE={rmse:.1f}, MAE={mae:.1f}")

        results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
        results_df.to_csv("results.csv", index=False)
        print("\nРезультаты моделей:\n", results_df)
        logger.info("5. Результаты сохранены")

    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
