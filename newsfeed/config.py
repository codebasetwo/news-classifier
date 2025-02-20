# flake8: noqa: E501
import logging
import logging.config
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import numpy as np
from hyperopt import hp
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import mlflow

STOPWORDS = stopwords.words("english")
STEMMER = SnowballStemmer("english")

# Root Directory
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Data Directories
DATASET_BASE_PATH = Path(ROOT_DIR, "datasets")
DATASET_BASE_PATH.mkdir(parents=True, exist_ok=True)
ZIP_LOC = Path(DATASET_BASE_PATH, "news-category-dataset.zip")

# Config Mlflow
MLFLOW_DIR = Path(
    ROOT_DIR, f"mlflow/{os.environ.get('GITHUB_USERNAME', 'mlflow_runs')}"
)
Path(MLFLOW_DIR).mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file://" + str(MLFLOW_DIR.absolute())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Metadata directory
METADATA_DIR = Path(ROOT_DIR, "datasets/metadata")

# Logging directory
LOGS_DIR = Path("/tmp/", "news-logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Hyperparameter search space
SPACE = {
    "learning_rate": hp.loguniform(
        "learning_rate", np.log(1e-5), np.log(10e-5)
    ),
    "num_epochs": hp.choice("num_epochs", [3, 5, 7, 10]),
    "batch_size": hp.choice("batch_size", [16, 32, 64]),
    "max_len": hp.choice("max_len", [128, 256]),
}

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
