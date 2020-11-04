import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# считаем набор данных, выкинем ненужные столбцы
df = pd.read_csv("heart_info.csv", sep=";")
df.drop(["bpm", "last_bpm", "Action", "Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.1.1", "Unnamed: 0.1.1.1"], axis=1, inplace=True)

# создадим обучающую и проверочную выборку
train_df = df[:550]
train_df.drop(["blood_volume"], axis=1, inplace=True)
X_train = np.asarray(train_df)
y_train = np.asarray(df[:550]["blood_volume"]).reshape((-1, ))

valid_df = df[550:]
valid_df.drop(["blood_volume"], axis=1, inplace=True)
X_valid = np.asarray(valid_df)
y_valid = np.asarray(df[550:]["blood_volume"]).reshape((-1, ))


def fit_and_evaluate(model): 
    model.fit(X_train, y_train)
    model_pred = model.predict(X_valid)
    model_acc = np.mean(model_pred - y_valid)
    return model_acc


# инициализируем и обучим модель
predictor = RandomForestRegressor()
fit_and_evaluate(predictor)


