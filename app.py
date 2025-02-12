from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from income_prediction.pipline.prediction_pipeline import PredictionPipeline
from income_prediction .constants import APP_HOST, APP_PORT
from income_prediction.pipline.training_pipeline import TrainPipeline
from income_prediction.constants import DEFAULT_PREDDICT_FILE_PATH, DEFAULT_PREDDICT_FILE_NAME
from fastapi.responses import JSONResponse
import traceback 
import pandas as pd
import tempfile
import io
import os
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "index.html",{"request": request, "context": "Rendering"})

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        # ✅ Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))  # Convert CSV to DataFrame

        print("Received DataFrame:\n")

        obj = PredictionPipeline()
        df = obj.initiate_prediction_pipeline(df)
        if df is not None:
            with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix='.csv') as temp_file:
                df.to_csv(temp_file, index=False)
                temp_file_path = temp_file.name  # Save the file path for returning

            # Return the processed CSV as a downloadable file
            return FileResponse(temp_file_path, media_type='text/csv', filename="processed_data.csv")
        else:
            print("Prediction failed")
            return JSONResponse(content={"status": False, "message": "Prediction failed"}, status_code=400)

    except Exception as e:
        return {"status": False, "error": str(e)}

    
@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
    
        df = pd.read_csv(os.path.join(DEFAULT_PREDDICT_FILE_PATH, DEFAULT_PREDDICT_FILE_NAME))
        obj = PredictionPipeline()
        df = obj.initiate_prediction_pipeline(df)
        if df is not None:
            with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix='.csv') as temp_file:
                df.to_csv(temp_file, index=False)
                temp_file_path = temp_file.name  # Save the file path for returning

            # Return the processed CSV as a downloadable file
            return FileResponse(temp_file_path, media_type='text/csv', filename="processed_data.csv")

        else:
            print("Prediction failed")
            return JSONResponse(content={"status": False, "message": "Prediction failed"}, status_code=400)

    except Exception as e:
        print("Error Traceback:", traceback.format_exc())  # ✅ Debugging
        return {"status": False, "message": "Prediction failed"}

@app.get("/train")
async def trainRouteClient():
    try:
        print("Training started")
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)