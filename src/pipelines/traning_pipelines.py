import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion

if __name__ == '__main__':
    object = DataIngestion()
    
    train_data_path,test_data_path = object.intial_data_ingestion()
    print(train_data_path,test_data_path)