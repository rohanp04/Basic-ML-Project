import os
import numpy as np
import pandas as pd
from src.exception import CustomException
import sys
import dill
from sklearn.metrics import r2_score


def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path,exist_ok=True)

        with open(filepath, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train) 

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)

            test_r2 = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_r2
        
        return report

    except:
        pass