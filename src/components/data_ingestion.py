import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

# Data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('./artifacts', 'train.csv')
    test_data_path: str = os.path.join('./artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion beginning")
        try:

            # os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            # df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Saving Train/Test datasets')
            train_data = pd.read_csv(os.path.join('../notebooks/data', 'train.csv'))
            test_data = pd.read_csv(os.path.join('../notebooks/data', 'test.csv'))

            categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race',
                                    'native-country']
            logging.info("Converting rare categories to a single category 'Other'")
            for feature in categorical_features:
                temp = train_data.groupby(feature)['result'].count() / len(train_data)
                temp_data = temp[temp > 0.01].index
                train_data[feature] = np.where(train_data[feature].isin(temp_data), train_data[feature], 'Other')

            for feature in categorical_features:
                temp = test_data.groupby(feature)['result'].count() / len(test_data)
                temp_data = temp[temp > 0.01].index
                test_data[feature] = np.where(test_data[feature].isin(temp_data), test_data[feature], 'Other')

                # one-hot encoding sex feature

            train_data['Gender'] = pd.get_dummies(train_data['sex'], drop_first=True)
            train_data = train_data.drop(columns=['sex'], axis=0)

            test_data['Gender'] = pd.get_dummies(test_data['sex'], drop_first=True)
            test_data = test_data.drop(columns=['sex'], axis=0)

            # Replacing . with null character in test_data target feature
            test_data['result'] = test_data['result'].str.replace('.', '')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion completed successfully!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error has occured at Data Ingestion stage")
            raise CustomException(e, sys)


# if __name__ == '__main__':
#     obj = DataIngestion()
#     train_data_path, test_data_path = obj.initiate_data_ingestion()
#     data_transformation = DataTransformation()
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
