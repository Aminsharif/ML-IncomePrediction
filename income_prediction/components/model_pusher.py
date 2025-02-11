import sys

from income_prediction.cloud_storage.aws_storage import SimpleStorageService
from income_prediction.exception import ExceptionHandle
from income_prediction.logger import logging
from income_prediction.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact,LocalModelPusherArtifact,LocalModelEvaluationArtifact
from income_prediction.entity.config_entity import ModelPusherConfig, LocalModelPusherConfig
from income_prediction.entity.s3_estimator import IncomePredictionEstimator,IncomePredictionLocalModelEstimator
from income_prediction.constants import *
import joblib

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.IncomePrediction_estimator = IncomePredictionEstimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=model_pusher_config.s3_model_key_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            logging.info("Uploading artifacts folder to s3 bucket")

            self.IncomePrediction_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)


            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise ExceptionHandle(e, sys) from e
        

class LocalModelPusher:
    def __init__(self, model_evaluation_artifact: LocalModelEvaluationArtifact,
                 model_pusher_config: LocalModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.IncomePrediction_estimator = IncomePredictionLocalModelEstimator(model_path=model_pusher_config.model_key_path)

    def initiate_model_pusher(self) -> LocalModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            logging.info("Uploading artifacts folder to s3 bucket")
            model = joblib.load(self.model_evaluation_artifact.trained_model_path)
            self.IncomePrediction_estimator.save_model(model)

            model_pusher_artifact = LocalModelPusherArtifact(model_path=self.model_pusher_config.model_key_path)

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise ExceptionHandle(e, sys) from e