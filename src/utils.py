import os
import numpy as np
import pandas as pd
from src.exception import CustomException
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path,exist_ok=True)

        with open(filepath, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i, model_name in enumerate(models.keys()):
            model = models[model_name]
            param = params.get(model_name, {})

            gs = GridSearchCV(model, param, cv=3, n_jobs=-1, scoring='r2')
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) 

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            report[model_name] = test_r2
        
        return report

    except Exception as e:
        print(f"Error in model evaluation: {e}")
        return None 