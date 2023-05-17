import os
import sys
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception has occured in utils.save_object")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pd.read_pickle(file_obj)

    except Exception as e:
        logging.info("Exception has occured in utils.load_object")
        raise CustomException(e, sys)


def evaluate_model(y_test, y_hat):
    accuracy = accuracy_score(y_test, y_hat)
    return accuracy

