from sklearn.impute import SimpleImputer  #Handing missing value
from sklearn.preprocessing import StandardScaler  #Handling feature scaling
from sklearn.preprocessing import OneHotEncoder   # Ordinal Encoding

## Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import numpy as np 
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

## Data transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')

## Data Transformation class
class Datatransformation:
    def __init__(self):
        self.data_transformation = DataTransformationconfig()
        
    def get_data_transformation_objest(self):
        
        try: 
          logging.info("Data Transformation is intialtiate")
          
          # Define numerical and categorical columns
          numerical_columns  = ['carat', 'depth', 'table', 'x', 'y', 'z']
          categorical_columns = ['cut', 'color', 'clarity']
          
         ## Defining the custom ranking for each of the ordinal variable
          cut_categories = ["Fair","Good","Very Good","Premium","Ideal"]
          color_categoty = ["D","E","F","G","H","I","J"]
          clarity_categories = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
          
          logging.info("Pipelines Intiated")
          
          ## Numercial Pipeline
          numerical_pipeline = Pipeline(
          steps=[
              ("imputer",SimpleImputer(strategy='median')),
              ("scaler",StandardScaler(with_mean=False))
             ]
          )

          ## Categorical Pipeline
          categorical_pipeline = Pipeline(
           steps=[
             ("imputer",SimpleImputer(strategy='most_frequent')),
             ('ordinalEncoder',OneHotEncoder(categories=[cut_categories,color_categoty,clarity_categories])),
             ('scaler',StandardScaler(with_mean=False))
            ]
          )

          preprocessor = ColumnTransformer([
            ("numerical_pipeline",numerical_pipeline,numerical_columns),
            ('categorical_pipeline',categorical_pipeline,categorical_columns)
            ])
          
          logging.info("Pipelines completed")
          
          return preprocessor
          
          
        except Exception as e:
            logging.info(e)

        
    def intiate_data_transformation(self,train_path,test_path):
        
        try:
            
            ## Reading the train test data
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read trains test data completed")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")
            
            logging.info("Obtain preprocessor object")
            
            preprocessor_object = self.get_data_transformation_objest()
            
            target_col = "price"
            drop_cols = [target_col,'id']
            
            # Dividing the features into independent and dependent features
            
            input_feature_train_df = train_df.drop(columns=drop_cols,axis=1)
            target_feature_train_df = train_df[target_col]
            
            input_feature_test_df = train_df.drop(columns=drop_cols,axis=1)
            target_feature_test_df = train_df[target_col]
            
            
            # Apply the transfromation
            
            logging.info("Appling preprocessing on traning and test data")
            input_feature_train_arr = preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_object.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Saving pickle file")
            
            save_object (
                file_path=self.data_transformation.preprocessor_ob_file_path,
                obj=preprocessor_object
             )
            
            
            logging.info("Preprocessor pickel is created and saved")
            
            return(
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_ob_file_path
            )
            
            
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
            
        
        