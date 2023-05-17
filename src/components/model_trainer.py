from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model
from dataclasses import dataclass
import sys
import os

import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting input and target variables")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1:],
                test_array[:, :-1],
                test_array[:, -1:]
            )

            models = {
                'RandomForestClassifier': RandomForestClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(n_estimators=350, algorithm='SAMME.R'),
                'SVC': SVC(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'MLPClassifier': MLPClassifier(),
                'GaussianNB': GaussianNB()
            }
            # dictionary for storing model reports
            model_score = {}
            print("Please be patient...")

            for i in range(len(list(models))):
                model = list(models.values())[i]
                progress = 'Training: ' + str(round((1 / len(list(models)) * 100) * (i + 1))) + '% complete'
                model.fit(X_train, y_train.ravel())
                print(progress, end='\r')
                y_hat = model.predict(X_test)
                accuracy = evaluate_model(y_test, y_hat)

                model_score[list(models.keys())[i]] = {'accuracy': accuracy}

            accuracy_dict = {}
            for model in model_score:
                accuracy_dict[model] = model_score[model]['accuracy'] * 100

            best_model = sorted(accuracy_dict.items(), key=lambda x:x[1], reverse=True)[0]
            print(f'Best Model Found! , Model name: {best_model[0]}, Accuracy: {best_model[1]}')
            print('='*40)
            logging.info(f'Best Model Found! , Model name: {best_model[0]}, Accuracy: {best_model[1]}')

            # Saving the pickle of the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models[best_model[0]]
            )

        except Exception as e:
            logging.info("Error occured in model_trainer.ModelTrainer")
            raise CustomException(e, sys)


