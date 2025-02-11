from income_prediction.entity.config_entity import ModelEvaluationConfig, LocalModelEvaluationConfig, DataTransformationConfig
from income_prediction.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact,LocalModelEvaluationArtifact
from sklearn.metrics import f1_score
from income_prediction.exception import ExceptionHandle
from income_prediction.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from income_prediction.logger import logging
import sys
import pandas as pd
from typing import Optional
from income_prediction.entity.s3_estimator import IncomePredictionEstimator,IncomePredictionLocalModelEstimator
from dataclasses import dataclass
from income_prediction.entity.estimator import TargetValueMapping
import joblib
import numpy as np
from income_prediction.utils.main_utils import read_yaml_file, drop_columns
from income_prediction.components.data_transformation import DataTransformation

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                          data_transformation_config = DataTransformationConfig())
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def get_best_model(self) -> Optional[IncomePredictionEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            IncomePrediction_estimator = IncomePredictionEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if IncomePrediction_estimator.is_model_present(model_path=model_path):
                return IncomePrediction_estimator
            return None
        except Exception as e:
            raise  ExceptionHandle(e,sys)
        

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

            target_test_df = test_df[TARGET_COLUMN]
            drop_cols = self._schema_config['drop_columns']

            feature_test_df = drop_columns(df=feature_test_df, cols = drop_cols)

            feature_test_df.replace('?',np.NaN,inplace=True)

            is_null_present_test_df, cols_with_missing_values_test_df = self.data_transformation.is_null_present(feature_test_df)
            if is_null_present_test_df:
                feature_test_df = self.data_transformation.impute_missing_values(feature_test_df, cols_with_missing_values_test_df)

            x, y = feature_test_df, target_test_df
            y = y.replace(
                TargetValueMapping()._asdict()
            )

            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score=None
            best_model = self.get_best_model() 
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise ExceptionHandle(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise ExceptionHandle(e, sys) from e
        

    def initiate_local_model_evaluation(self) -> LocalModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = LocalModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise ExceptionHandle(e, sys) from e       


class LocalModelEvaluation:

    def __init__(self, model_eval_config: LocalModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                          data_transformation_config = DataTransformationConfig())
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def get_best_model(self) -> Optional[IncomePredictionLocalModelEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            model_path=self.model_eval_config.local_model_path
            IncomePrediction_estimator = IncomePredictionLocalModelEstimator(model_path=model_path)

            if IncomePrediction_estimator.is_model_present(model_path=model_path):
                return IncomePrediction_estimator

            return None
        except Exception as e:
            raise  ExceptionHandle(e,sys)


    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

            target_test_df = test_df[TARGET_COLUMN]
            drop_cols = self._schema_config['drop_columns']

            feature_test_df = drop_columns(df=feature_test_df, cols = drop_cols)

            feature_test_df.replace('?',np.NaN,inplace=True)

            is_null_present_test_df, cols_with_missing_values_test_df = self.data_transformation.is_null_present(feature_test_df)
            if is_null_present_test_df:
                feature_test_df = self.data_transformation.impute_missing_values(feature_test_df, cols_with_missing_values_test_df)

            x, y = feature_test_df, target_test_df
            y = y.replace(
                TargetValueMapping()._asdict()
            )

            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score=None
            best_model = self.get_best_model() # Commented out to  s3 access
            if best_model is not None:
                model = joblib.load(self.model_eval_config.local_model_path)
                y_hat_best_model = model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise ExceptionHandle(e, sys)

    def initiate_model_evaluation(self) -> LocalModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = LocalModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise ExceptionHandle(e, sys) from e