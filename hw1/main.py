import pandas as pd
import numpy as np
import warnings
import moduls.data_loader as data_loader
import moduls.data_visual as data_visual
import moduls.log as log

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
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
    logger.info(" === Запуск программы === ")

    try:
        # 1. Загрузка данных и создание БД
        df = data_loader.create_database(
            csv_path="data/bigmart_sales.csv",
            db_path="db/bigmart.db"
        )
        logger.info(f" 1. Данные успешно загружены: {df.shape}")

        # 2. Предобработка данных
        df = df.dropna(how='all')  # удалить полностью пустые строки

        # Заполнение пропусков для числовых и категориальных признаков
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                mode_val = df[col].mode(dropna=True)
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna("Unknown")

        # Специальная обработка для известных столбцов
        if 'Item_Weight' in df.columns:
            df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
        if 'Outlet_Size' in df.columns:
            df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

        # Приведение категорий
        if 'Item_Fat_Content' in df.columns:
            df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
                'LF': 'Low Fat',
                'low fat': 'Low Fat',
                'reg': 'Regular'
            })

        # Новые признаки
        if 'Outlet_Establishment_Year' in df.columns:
            df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']

        if 'Item_MRP' in df.columns:
            df['MRP_Tier'] = pd.cut(df['Item_MRP'], bins=[0, 100, 200, 300, 400], labels=['Low', 'Medium', 'High', 'Very High']).astype(str)
            df['MRP_Bin'] = pd.cut(df['Item_MRP'], bins=5, labels=False)

        if 'Item_Visibility' in df.columns and 'Item_Identifier' in df.columns:
            mean_visibility = df.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
            df['Visibility_MeanRatio'] = df['Item_Visibility'] / mean_visibility

        # Заменяем бесконечные и слишком большие значения
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        # Категориальные признаки
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Label Encoding для категорий
        label_encoders = {}
        for column in categorical_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        X = df.drop("Item_Outlet_Sales", axis=1)
        y = df["Item_Outlet_Sales"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f" 2. Предобратока окончена")

        # 3. Модели 
        
        # Базовые модели без оптимизации
        models = {
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso(alpha=0.1),
            "Ridge": Ridge(alpha=1.0),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_depth=10, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42),
        }

        # Оптимизация CatBoost
        def objective_catboost(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 300, 800),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_state": 42,
                "verbose": 0
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

        # Оптимизация RandomForest
        def objective_rf(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": 42
            }
            model = RandomForestRegressor(**params)
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, np.log1p(y_train), cv=cv, scoring="r2")
            return np.mean(scores)

        study_rf = optuna.create_study(direction="maximize")
        study_rf.optimize(objective_rf, n_trials=30)
        best_rf = RandomForestRegressor(**study_rf.best_params)
        best_rf.fit(X_train, np.log1p(y_train))
        models["RandomForest_Opt"] = best_rf
        logger.info(f" 3. Модели определены")

        # 4. Обучение и метрики 
        results = []
        for name, model in models.items():
            logger.info(f" Обучение модели: {name}")

            y_train_log = np.log1p(y_train)
            y_test_log = np.log1p(y_test)
            
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
            
            # Для линейных моделей делаем коррекцию смещения
            if name in ["LinearRegression", "Lasso", "Ridge"]:
                residuals = y_train_log - model.predict(X_train)
                sigma2 = np.var(residuals)
                y_pred = np.expm1(y_pred_log + sigma2 / 2)
            else:
                y_pred = np.expm1(y_pred_log)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            logger.info(f"  {name} → R2={r2:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

        logger.info(f" 4. Модели обучены")

        # 5. Таблица с результатами 
        results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
        results_df.to_csv("results.csv", index=False)

        print("\nРезультаты моделей:\n", results_df)
        logger.info(" 5. Результаты моделей сохранены отдельно в results.csv")
        
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
