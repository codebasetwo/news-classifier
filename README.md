<div align="center">
<h1> <img width="30" src="https://www.google.com/imgres?q=news%20logo&imgurl=https%3A%2F%2Fstatic.vecteezy.com%2Fsystem%2Fresources%2Fpreviews%2F007%2F539%2F914%2Fnon_2x%2Fnews-logo-design-vector.jpg">&nbsp;NEWS CLASSIFIER</h1>
Classify news articles into different categories
</div>

<br>

<div align="center">
    <a target="_blank" href="https://www.linkedin.com/in/emekan"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/codebasetwo"><img src="https://img.shields.io/twitter/follow/codebasetwo.svg?label=Follow&style=social"></a>
</div>

<br>
<hr>

## Overview
Plagued with the plethora of content accross the internet most of which are unimportant to certain users, this project aims at cutting down time individuals spend in engaging with news articles or topics accross the noisy internet, that are not of interest to them. By discovering news content from various popular and trusted news channels, we can easily categorize them by using this ML service.
<br>

This Project is an end-end machine learning project with various component:
- **Iterate:** Continously iterate on model and data
- **Scale:** Scale service as per traffic
- **CI/CD:** To continously train and and deploy better models.
- **MLOps:** Connecting software engineering best principles to machine learning workflows MLOps

## Data
My dataset was gotten from [Kaggle](https://www.kaggle.com), so it was freely available. Therefore was no cost in getting data.

## Set up
All workload was carried out on my personal laptop. `Training` and `Tuning` was done in google colab. <br>
1. `Download` and `Install` [miniconda](https://docs.anaconda.com/miniconda/install/)


2. **Create Virtual environment**

    ```bash
    conda create -n news-classifier python=3.11
    conda activate news-classifier
    ```
    ```bash 
    git clone https://github.com/codebasetwo/news-classifier.git .
    cd news-classifier
    export PYTHONPATH=$PYTHONPATH:$PWD
    pip install -r requirements.txt
    pre-commit install
    pre-commit autoupdate
    ```
3. ***Credentials***

    ```bash
    # create environment file
    touch .env
    ```
    ```bash
    # In the .env file setup any needed credentials
    GITHUB_USERNAME="CHANGE_THIS_TO_YOUR_USERNAME"  # ← CHANGE THIS
    GEMINI_API_KEY="CHANGE_THIS_TO_YOUR_GEMINI_KEY" # ← CHANGE THIS
    # in your .bashrc file linux users you can also add this at the end to access your Gemini key best practice
    export GEMINI_API_KEY="CHANGE_THIS_TO_YOUR_GEMINI_KEY" # ← CHANGE THIS
    ```

    ```bash
    source .env
    ```

## Notebook

Start by exploring the [jupyter notebook](notebooks/newsfeed.ipynb) to interactively walkthrough the machine learning workflow.

```bash
  # Start notebook
  jupyter lab notebooks/newsfeed.ipynb
```

## Scripts

We can also execute the same workloads in the notebooks using the clean Python scripts following software engineering best practices (training, tuning, testing, documentation, serving, versioning, etc.) `Caveat` since notebooks are mainly for iteration the code in the scripts might look more robust and better fornatted. The codes implemented in the notebook was refactored into the following scripts:

```bash
newsfeed
├── config.py
├── data.py
├── evaluate.py
├── models.py
├── plot.py
├── predict.py
├── serve.py
├── train.py
├── tune.py
└── utils.py
```


### Training
```bash
export EXPERIMENT_NAME="news_model_experiment"
export DATASET_LOC="/datasets/train.csv"
export PARAMS='{"num_epochs": 3, "max_length": 128, "batch_size": 64, "learning_rate": 5e-5}'
python newsfeed/train.py \
    --dataset-loc "$DATASET_LOC" \
    --params "$PARAMS" \
    --experiment-name "$EXPERIMENT_NAME" \

```

### Tuning
```bash
export EXPERIMENT_NAME="news_model_experiment"
export DATASET_LOC="/datasets/train.csv"
python newsfeed/train.py \
    --dataset-loc "$DATASET_LOC" \
    --experiment-name "$EXPERIMENT_NAME" \
```

### Experiment tracking

I used [MLflow](https://mlflow.org/) to track our experiments and store our models and the [MLflow Tracking UI](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) to view our experiments. [MLflow](https://mlflow.org/) helps to have a central location to store all of our experiments. It's easy and inexpensive to spin up it is also open source so can be used freely they are other managed solutions. 

```bash
export MODEL_REGISTRY=$(python -c "from newsfeed import config; print(config.MODEL_REGISTRY)")
mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri $MODEL_REGISTRY --default-artifact-root $MODEL_REGISTRY
```
You can go to  <a href="http://localhost:8080/" target="_blank">http://localhost:8080/</a> to view your MLflow dashboard.

</details>


### Evaluation
```bash
export EXPERIMENT_NAME="news_model_experiment"
export RESULTS_FILE=results/evaluation_results.json
export HOLDOUT_LOC="datasets/holdout.csv"
python newsfeed/evaluate.py \
    --experiment_name "$EXPRIMENT_NAME" \
    --test_file_path "$HOLDOUT_LOC" \
    --results_file_path "$RESULTS_FILE" \
```
```json
{
     {
        "overall": {
            "precision": ...,
            "recall": ...,
            "f1_score": ...,
            "accuracy": ...,
    },
        "per_class": {...},
        "slices": {...},
    }
}
...

```

### Inference
```bash
export EXPERIMENT_NAME="news_model_experiment"
python newsfeed/predict.py predict \
    --experiment_name $EXPERIMENT_NAME \
    --metric "accuracy" \
    --mode "DESC" \
```

```json
[{
  "prediction": [
    "NEWS & POLITICS"
  ],
  "probabilities": {
    "NEWS & POLITICS": 0.969523921,
    "PARENTING": 0.052329571,
    "ENTERTAINMENT": 0.278275612,
    "ARTS, CULTURE & TRAVEL": 0.084231049,
    "EDUCATION": 0.061039911,
    "SPORTS & WELLNESS": 0.050200002
  }
}]
```

### Serving

  ```bash
  # Set up
  export EXPERIMENT_NAME="news_model_experiment"
  python newsfeed/serve.py --experiment_name $EXPERIMENT_NAME
  ```

  Once the application is running, we can use it via CURL, Python, etc.: <br>
  1. `Via CURL` <br>
      ```bash
      curl -X POST http://localhost:80/predict \
          -H "Content-Type: application/json"
      curl -X POST  http://127.0.0.1:8000/predict/predict/ \
      -H "Content-Type: application/json" \
      -d '[
              {
                  "link": "https://www.huffingtonpost.com/entry/prescription-drug-overdose_us_5b9d7ea7e4b03a1dcc88b84f",
                  "headline": 'Texas Court OKs "Up Skirts": WTF?!',
                  "category": "NEWS & POLITICS",
                  "short_description": "This week, the Texas Court of Appeals made a ruling that is both outrageous and grotesque... and for the first time in recorded history, it has nothing to do with either abortion or the death penalty.",
                  "authors": "Jon Hotchkiss, ContributorHost, Be Less Stupid",
                  "date": "2014-09-23",
              }
              ]'
      ```
  2. `Via Python` <br>
      ```python
      # Python
      import json
      import requests
      link = "https://www.huffingtonpost.com/entry/prescription-drug-overdose_us_5b9d7ea7e4b03a1dcc88b84f"
      headline = 'Texas Court OKs "Up Skirts": WTF?!'
      category: "NEWS & POLITICS"
      short_description = "This week, the Texas Court of Appeals made a ruling that is both outrageous and          grotesque... and for the first time in recorded history, it has nothing to do with either abortion or the death penalty."
      authors = "Jon Hotchkiss, ContributorHost, Be Less Stupid"
      date = "2014-09-23"
      data = [{"link": link, "headline": headline, "category": category, 
      "short_description": short_description, "authors": authors, "date": date}]
      requests.post("http://127.0.0.1:8000/predict", data=data).json()
      ```

### Testing
```bash
export EXPERIMENT_NAME="news_model_experiment"
HOLDOUT_LOC="datasets/holdout.csv"
# Test data
echo "Running tests for data..."
pytest --dataset-loc=$HOLDOUT_LOC tests/data --verbose --disable-warnings | tee 

# Test code
echo "Running tests for code..."
pytest --experiment-name=$EXPERIMENT_NAME tests/code --verbose --disable-warnings | tee 

# Test model
echo "Running tests for models..."
pytest --experiment-name=$EXPERIMENT_NAME tests/models --verbose --disable-warnings | tee
```