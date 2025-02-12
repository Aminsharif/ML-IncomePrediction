from income_prediction.entity.config_entity import (DataValidationConfig, DataTransformationConfig)
import sys
import numpy as np
import pandas as pd
from income_prediction.constants import SCHEMA_FILE_PATH, MODEL_FILE_NAME, MODEL_DIR
from income_prediction.entity.config_entity import DataTransformationConfig
from income_prediction.exception import ExceptionHandle
from income_prediction.logger import logging
from income_prediction.utils.main_utils import read_yaml_file, drop_columns
from income_prediction.components.data_transformation import DataTransformation
import joblib
import os
from pandas import DataFrame
from income_prediction.entity.estimator import TargetValueMapping

class Prediction:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.data_transformation = DataTransformation()
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

    def ModelPrediction(self, df: DataFrame) -> DataFrame:
        try:
            drop_cols = self._schema_config['drop_columns']

            feature_df = drop_columns(df=df, cols = drop_cols)
            feature_df=feature_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            feature_df.replace('?',np.NaN,inplace=True)

            is_null_present_df, cols_with_missing_values_df = self.data_transformation.is_null_present(feature_df)
            if is_null_present_df:
                feature_df = self.data_transformation.impute_missing_values(feature_df, cols_with_missing_values_df)

            x = feature_df
    
            model = joblib.load(os.path.join(MODEL_DIR, MODEL_FILE_NAME))
            prediction_values = model.predict(x)
           
            df['prediction'] = prediction_values

            # Replace the encoded prediction values with their original labels using reverse mapping
            target_value_mapping = TargetValueMapping()
            df['prediction'] = df['prediction'].replace(target_value_mapping.reverse_mapping())
            
            logging.info(f"Result: {prediction_values}")
            return df

        except Exception as e:
            raise ExceptionHandle(e, sys)
        
    def PredictDataValidation(self, df: DataFrame) -> bool:
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            # df =  pd.read_csv(os.path.join('default_predict_file', 'default.csv'))

            status =self.validate_number_of_columns(dataframe=df)

            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=df)

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0

            if validation_status:

                logging.info(f"Data validation successful")
                return True

            else:
                logging.info(f"Validation_error: {validation_error_msg}")
                return False
        except Exception as e:
            raise ExceptionHandle(e, sys) from e
        

    
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["feature_columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise ExceptionHandle(e, sys)
    
    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise ExceptionHandle(e, sys) from e
    