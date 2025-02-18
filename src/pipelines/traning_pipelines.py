import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Datatransformation
from src.components.model_tranier import ModelTrainer

if __name__ == '__main__':
    object = DataIngestion()
    
    train_data_path,test_data_path = object.intial_data_ingestion()
    print(train_data_path,test_data_path)
    
    data_transformation = Datatransformation()
    train_arr,test_arr,preprocessor_path = data_transformation.intiate_data_transformation(train_data_path,test_data_path)
    
    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)