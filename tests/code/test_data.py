import pandas as pd
import pytest

from newsfeed import data, config


@pytest.fixture(scope="module")
def stemmer():
    return config.STEMMER


@pytest.fixture(scope="module")
def class_to_index():
    return {"NEWS & POLITICS": 0, "ENTERTAINMENT": 1}


@pytest.fixture(scope="function")
def df():
    return pd.DataFrame(
        [
            {
                "link": "https://www.huffingtonpost.com/entry/prescription-drug-overdose_us_5b9d7ea7e4b03a1dcc88b84f",
                "headline": 'Texas Court OKs "Up Skirts": WTF?!',
                "category": "NEWS & POLITICS",
                "short_description": "This week, the Texas Court of Appeals made a ruling that is both outrageous and grotesque... and for the first time in recorded history, it has nothing to do with either abortion or the death penalty.",
                "authors": "Jon Hotchkiss, ContributorHost, Be Less Stupid",
                "date": "2014-09-23",
            }
        ])


def test_load_dataframe(dataset_loc):
    num_samples = 100
    dataset =  data.load_dataframe(dataset_loc, 100)
    assert len(dataset) == num_samples


def test_split_dataframe():
    num_per_Class = 10
    targets = ["Jane Doe"] * num_per_Class + ["John Doe"] * num_per_Class
    df = pd.DataFrame({"targets": targets, "ids":list(range(20))})
    df1, df2 = data.split_train_test(df=df, strat_cols="targets", test_size=0.5)
    assert df1.targets.value_counts().to_dict() == df2.targets.value_counts().to_dict()


@pytest.mark.parametrize(
    "text, stopword, expected",
    [
        ('hi', [], 'hi'),
        ("you're", ["you're"], ''),
        ("targeted", [], "target"),
        ("having", [], "have"),
    ]

)
def test_clean_text(text, stopword, expected, stemmer):
    assert data.clean_text(text=text, stopwords=stopword, stemmer=stemmer) == expected


def test_preprocess(df, class_to_index):
    max_length = 128
    output = data.preprocess(df=df, class_index=class_to_index, max_length=max_length)
    assert len(output) == 2
    assert list(output[0].keys()) == ["input_ids", "attention_mask"]
    assert output[0]["input_ids"].shape[1] == max_length
    assert output[0]["attention_mask"].shape[1] == max_length
    assert output[1][0] == 0


def test_preprocessor(df, class_to_index):
    preprocessor = data.DataPreprocessor(df=df, class_to_index=class_to_index)
    output = preprocessor.transform()
    assert len(output) == 3