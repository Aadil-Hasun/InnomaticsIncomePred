import sys
import os

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('src/pipeline/artifacts', 'preprocessor.pkl')
            model_path = os.path.join('src/pipeline/artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)

            model = load_object(model_path)

            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)

            return pred


        except Exception as e:
            logging.info("Exception has occured in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 age:float,
                 workclass:str,
                 fnlwgt:float,
                 education_num:int,
                 marital_status:str,
                 occupation:str,
                 relationship:str,
                 race:str,
                 capital_gain:float,
                 capital_loss:float,
                 hours_per_week:float,
                 native_country:str,
                 Gender:int):
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country
        self.Gender = Gender

    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                'age': [self.age],
                'workclass': [self.workclass],
                'fnlwgt': [self.fnlwgt],
                'education-num': [self.education_num],
                'marital-status': [self.marital_status],
                'occupation': [self.occupation],
                'relationship': [self.relationship],
                'race': [self.race],
                'capital-gain': [self.capital_gain],
                'capital-loss': [self.capital_loss],
                'hours-per-week': [self.hours_per_week],
                'native-country': [self.native_country],
                'Gender': [self.Gender]
            }
            df = pd.DataFrame(custom_data_dict)
            logging.info("CustomData DataFrame created")
            return df

        except Exception as e:
            logging.info("Exception has occured in prediction pipeline (CustomData)")
            raise CustomException(e, sys)


