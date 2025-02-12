# from income_prediction.pipline.training_pipeline import TrainPipeline


# obj = TrainPipeline()
# obj.run_pipeline()

from income_prediction.pipline.prediction_pipeline import PredictionPipeline
import pandas as pd
import os
obj = PredictionPipeline()
df = pd.read_csv(os.path.join('default_predict_file', 'default.csv'))
res = obj.initiate_prediction_pipeline(df)
if res is not None:
    print(res['prediction'])
else:
    print("Prediction failed")
