import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_filepath = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation initiated")

            categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
            numerical_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
            discrete_features = ['education-num', 'Gender']

       #      workclass_categories = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
       # 'Local-gov', 'Self-emp-inc', 'Other']
       #      marital_categories = ['Never-married', 'Married-civ-spouse', 'Divorced',
       # 'Married-spouse-absent', 'Separated', 'Other', 'Widowed']
       #      occupation_categories = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
       # 'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
       # 'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
       # 'Craft-repair', 'Protective-serv', 'Other']
       #      relationship_categories = ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
       # 'Other-relative']
       #      race_categories = ['White', 'Black', 'Asian-Pac-Islander', 'Other']
       #      country_categories = ['United-States', 'Other', 'Mexico']

            # preprocessor pipelines
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    # ('encoder', LabelEncoder(classes_=[workclass_categories, marital_categories, occupation_categories, relationship_categories, race_categories, country_categories])),
                    ('encoder', BinaryEncoder()),
                    ('scaler', StandardScaler())
                ]
            )
            dis_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('scaler', StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features),
                ('dis_pipeline', dis_pipeline, discrete_features)
            ])

            logging.info("Creation of data transformation pipeline completed")
            return preprocessor

        except Exception as e:
            logging.info("Error has occured in Data Transformation stage")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data files as pandas DataFrame
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Reading of train and test data completed.")

            # X_train, y_train, X_test, y_test split
            X_train = train_data.drop(columns=['result'], axis=0)
            y_train = train_data[['result']]

            X_test = test_data.drop(columns=['result'], axis=0)
            y_test = test_data[['result']]

            # Obtaining preprocessing objects
            logging.info('Obtaining preprocessing objects')
            preprocessor = self.get_data_transformation_obj()

            # Applying preprocessing
            logging.info("Applying preprocessing on train and test data")
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            logging.info("Preprocessing completed")

            # Concatenating X_train and y_train into a single numpy array
            train_arr = np.c_[X_train, y_train]
            # Concatenating X_test and y_test into a single numpy array
            test_arr = np.c_[X_test, y_test]

            # Saving the preprocessor objects
            save_object(file_path=self.data_transformation_config.preprocessor_filepath,
                        obj=preprocessor
                        )

            logging.info("Preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_filepath
            )

        except Exception as e:
            logging.info("Exception has occured in the initiate_data_transformation function")
            raise CustomException(e, sys)



