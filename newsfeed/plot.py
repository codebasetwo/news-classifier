# flake8: noqa: E501
import warnings
from typing import Dict

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()
from IPython.display import display
from ipywidgets import widgets
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from wordcloud import STOPWORDS, WordCloud

from utils import load_json
from config import METADATA_DIR


def plot(df: pd.DataFrame, category: str) -> None:
    """
    Plots a word cloud for the given category.

    Args:
        category: The tag for which to generate the word cloud.
    """
    plt.close()

    subset = df[df.category == category]
    text = subset.headline.values
    cloud = WordCloud(
        stopwords=STOPWORDS,
        background_color="black",
        collocations=False,
        width=500,
        height=300,
    ).generate(" ".join(text))
    plt.figure(figsize=(10, 3))
    plt.axis("off")
    plt.imshow(cloud)
    plt.show()


def create_category_selector(df: pd.DataFrame):
    """
    Creates a widget for category selection.

    Args:
        df : DataFrame, the dataset containing the category information.
    """
    categories = df["category"].unique()  # Get unique tags from the DataFrame
    return widgets.Dropdown(
        options=categories, value=categories[0], description="Category:"
    )


# categories =  df['category'].unique() # Get unique tags from your DataFrame
# cat_selector = widgets.Dropdown( options=categories, value=categories[0], description='Category:' )

# # Display the widget and output
# output = widgets.Output()


def on_change(change: Dict, df: pd.DataFrame, output):
    """
    Function to handle changes in the category selector.

    Args:
        change : dict, the change event from the widget.
        df     : DataFrame, the dataset containing the headlines.
        output : Output widget, the output widget to display the word cloud.
    """
    with output:
        output.clear_output(wait=True)
        plot(df, change["new"])


def plot_wordcloud(df: pd.DataFame):
    """
    Sets up the word cloud plot with the category selector and output widget.

    Args:
        df : DataFrame, the dataset containing the headlines.
    """
    output = widgets.Output()
    cat_selector = create_category_selector(df)
    cat_selector.observe(
        lambda change: on_change(change, df, output), names="value"
    )
    display(cat_selector, output)


# Generate the confusion matrix
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot confusion matrix of of model prediction against the true value.

    Args:
        y_true (array-like): array of true values.
        y_pred (array-like): array of predicted model predicted values.

    """

    metadata = "class_to_index.json"
    class_to_index = load_json(f"{METADATA_DIR}/{metadata}")

    index_to_class = {idx: cls for cls, idx in class_to_index.items()}
    y_true = [index_to_class[x] for x in y_true]
    y_pred = [index_to_class[x] for x in y_pred]
    labels = list(index_to_class.values())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=15, pad=20)
    plt.xlabel("Prediction", fontsize=11)
    plt.ylabel("Actual", fontsize=11)

    # Customizations
    plt.gca().xaxis.set_label_position("top")
    plt.gca().xaxis.tick_top()
    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, "Prediction", ha="center", fontsize=13)

    plt.show()
