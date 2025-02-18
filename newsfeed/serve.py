# flake8: noqa: E501
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import List, Optional

import evaluate
import predict
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

import mlflow
from config import METADATA_DIR, MLFLOW_TRACKING_URI

models = {}


def get_model(
    experiment_name: str = "news_model_experiment",
    metric: str = "accuracy",
    mode: str = "DESC",
):

    run_id = predict.get_best_run_id(
        experiment_name=experiment_name, metric=metric, mode=mode
    )
    global model

    model = predict.get_best_model(run_id=run_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["my_model"] = model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    yield
    models["my_model"].clear()


# Initialize FastAPI
app = FastAPI(title="Classify news headline", lifespan=lifespan)


class PredictionRequest(BaseModel):
    data: List[dict] = Field(
        ...,
        min_items=1,
        example=[
            {
                "link": "https://www.huffingtonpost.com/entry/prescription-drug-overdose_us_5b9d7ea7e4b03a1dcc88b84f",
                "headline": 'Texas Court OKs "Up Skirts": WTF?!',
                "category": "",
                "short_description": "This week, the Texas Court of Appeals made a ruling that is both outrageous and grotesque... and for the first time in recorded history, it has nothing to do with either abortion or the death penalty.",
                "authors": "Jon Hotchkiss, ContributorHost, Be Less Stupid",
                "date": "2014-09-23",
            }
        ],
        description="List of Dictionary(ies) for input",
    )


class EvaluationRequest(BaseModel):
    experiment_name: str = Field(..., example="news_exp")
    test_file_path: str = Field(..., example="/data/test.csv")
    metric: str = Field(default="accuracy")
    mode: str = Field(default="DESC")
    batch_size: int = Field(default=32, ge=1, le=256)
    num_samples: Optional[int] = Field(default=None)
    results_file_path: str = Field(
        default=str(METADATA_DIR / "evaluations.json")
    )
    save: bool = Field(default=False)


@app.get("/")
def home():
    """Health check"""

    print(
        "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."
    )
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }

    return response


@app.post("/predict")
async def _predict(request: PredictionRequest):
    results = predict.predict_proba(
        test_ds=request.data, model=models["my_model"]
    )
    return {"result": results}


@app.post("/evaluate")
async def _evaluate(request: EvaluationRequest):
    metrics = evaluate.evaluate(
        experiment_name=request.experiment_name,
        test_file_path=request.test_file_path,
        metric=request.metric,
        mode=request.mode,
        batch_size=request.batch_size,
        num_samples=request.num_samples,
        results_file_path=request.results_file_path,
        save=request.save,
    )
    return {"result": metrics}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Serve the model to get prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "eperiment_name", type=str, help="Name of the experiment"
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="metric to use for selecting model",
        default="accuracy",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="mode of sorting metrics",
        default="DESC",
        choices=["DESC", "ASC"],
    )

    args = parser.parse_args()

    get_model(
        experiment_name=args.experiment_name,
        metric=args.metric,
        mode=args.mode,
    )

    uvicorn.run(
        "serve:app", host="0.0.0.0", port=8000, reload=True, log_level="debug"
    )
