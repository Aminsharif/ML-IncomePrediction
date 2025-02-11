import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from income_prediction.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from income_prediction.entity.config_entity import DataTransformationConfig
from income_prediction.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from income_prediction.exception import ExceptionHandle
from income_prediction.logger import logging
from income_prediction.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from income_prediction.entity.estimator import TargetValueMapping
from sklearn.impute import SimpleImputer


class DataTransformation():
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact=None,
                 data_transformation_config: DataTransformationConfig=None,
                 data_validation_artifact: DataValidationArtifact=None):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise ExceptionHandle(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ExceptionHandle(e, sys)

    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initialize PowerTransformer")

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise ExceptionHandle(e, sys) from e
        

    def is_null_present(self,data):
        """
            Method Name: is_null_present
            Description: This method checks whether there are null values present in the pandas Dataframe or not.
            Output: Returns True if null values are present in the DataFrame, False if they are not present and
                returns the list of columns for which null values are present.
             On Failure: Raise Exception
        """
        logging.info('Entered the is_null_present method of the DataTransformation class')
        null_present = False
        cols_with_missing_values=[]
        cols = data.columns
        try:
            null_counts=data.isna().sum() # check for the count of null values per column
            for i in range(len(null_counts)):
                if null_counts[i]>0:
                    null_present=True
                    cols_with_missing_values.append(cols[i])
            if(null_present): # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
            logging.info('Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the DataTransformation class')
            return null_present, cols_with_missing_values
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def impute_missing_values(self, data, cols_with_missing_values):
        """
             Method Name: impute_missing_values
             Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
             Output: A Dataframe which has all the missing values imputed.
             On Failure: Raise Exception
        """

        logging.info(
            "Entered the impute_missing_values method of DataTransformation class"
        )

        try:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            for col in cols_with_missing_values:
                data[col] = imputer.fit_transform (data[[col]]).reshape(-1)
            logging.info('Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return data
        except Exception as e:
            raise ExceptionHandle(e, sys) from e
    
    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
                train_df=train_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                test_df=test_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

                train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].map({'<=50K' : 'less_equal_50K', '>50K' : "greater_50K"})
                test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].map({'<=50K' : 'less_equal_50K', '>50K' : "greater_50K"})
                
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]


                drop_cols = self._schema_config['drop_columns']

                logging.info("drop the columns in drop_cols of Training dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)

                input_feature_train_df.replace('?',np.NaN,inplace=True)
                
                is_null_present_train_df, cols_with_missing_values_train_df = self.is_null_present(input_feature_train_df)
                if is_null_present_train_df:
                    input_feature_train_df = self.impute_missing_values(input_feature_train_df, cols_with_missing_values_train_df)

                logging.info("Got train features and test features of Training dataset")
                
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )


                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

                target_feature_test_df = test_df[TARGET_COLUMN]

                input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)

                input_feature_test_df.replace('?',np.NaN,inplace=True)

                is_null_present_test_df, cols_with_missing_values_test_df = self.is_null_present(input_feature_test_df)
                if is_null_present_test_df:
                    input_feature_test_df = self.impute_missing_values(input_feature_test_df, cols_with_missing_values_test_df)

                logging.info("drop the columns in drop_cols of Test dataset")

                target_feature_test_df = target_feature_test_df.replace(
                TargetValueMapping()._asdict()
                )

                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Applying SMOTEENN on Training dataset")

                smt = SMOTEENN(sampling_strategy="minority")

                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )

                logging.info("Applied SMOTEENN on training dataset")

                logging.info("Applying SMOTEENN on testing dataset")

                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )

                logging.info("Applied SMOTEENN on testing dataset")

                logging.info("Created train array and test array")

                train_arr = np.c_[
                    input_feature_train_final.toarray(), np.array(target_feature_train_final)
                ]

                test_arr = np.c_[
                    input_feature_test_final.toarray(), np.array(target_feature_test_final)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise ExceptionHandle(e, sys) from e