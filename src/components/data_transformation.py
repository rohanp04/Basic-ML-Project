import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformation()

    def get_data_transformer_object(self):
        try:
            numerical_colums = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parential_level_of_education',
                'test_preparation_course'
            ]
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(stratergy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Categorical,numerical --> scaling,endocing done ")

            preprocessor = ColumnTransformer(
                [
                    ("num",num_pipeline,numerical_colums),
                    ("cat",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)



