# Newsfeed Classification Project

Welcome to the documentation for the **Newsfeed Classification Project**! This project focuses on building, training, and deploying a machine learning model to classify news articles into predefined categories. Below is an overview of the project structure and links to detailed documentation for each component.

---

## Project Overview

The goal of this project is to develop a robust and scalable system for classifying news articles into 6 categories which are "NEWS & POLITICS," "SPORTS & WELLNESS," "ARTS, CULTURE & TRAVEL," "EDUCATION," "PARENTING," "ENTERTAINMENT," . The project includes the following key components:

1. **Data Preparation**: Loading, preprocessing, and splitting the dataset.
2. **Model Training**: Training a deep learning model (e.g., BERT-based) on the preprocessed data.
3. **Hyperparameter Tuning**: Optimizing model hyperparameters using tools like Hyperopt.
4. **Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
5. **Prediction**: Generating predictions for new news articles.
6. **Model Serving**: Deploying the trained model for inference.
7. **Utilities**: Helper functions for data processing, logging, and more.

---

## Documentation Structure

The documentation is organized into the following sections:

- **[Configuration](newsfeed/config.md)**: Details about project configuration, including paths, hyperparameters, and environment setup.
- **[Data](newsfeed/data.md)**: Information about data loading, preprocessing, and splitting.
- **[Training](newsfeed/train.md)**: Documentation for model training, including loss functions, optimizers, and callbacks.
- **[Hyperparameter Tuning](newsfeed/tune.md)**: Guide to tuning model hyperparameters using Hyperopt.
- **[Prediction](newsfeed/predict.md)**: How to generate predictions using the trained model.
- **[Evaluation](newsfeed/evaluate.md)**: Metrics and methods for evaluating model performance.
- **[Model](newsfeed/models.md)**: Details about the model architecture and implementation.
- **[Serving](newsfeed/serve.md)**: Instructions for deploying the model for inference.
- **[Utilities](newsfeed/utils.md)**: Helper functions and utilities used throughout the project.

---

## Getting Started

To get started with the project, follow these steps:

1. **Set up the environment**: Install the required dependencies using `pip install -r requirements.txt`.
2. **Prepare the data**: Use the scripts in the `data/` directory to load and preprocess the dataset.
3. **Train the model**: Run the training script with your desired configuration.
4. **Tune hyperparameters**: Use the tuning script to optimize model performance.
5. **Evaluate the model**: Assess the model's performance on the test set.
6. **Deploy the model**: Serve the trained model using the provided serving script.

---

## Caveat

This project can also be done in a distributed manner but due to my resource constraint I was limited to this But all ideas can easily be translated to a  distributed manner. Training and tuning was done using google colab.

---

## Contributing

I welcome contributions to this project! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/codebasetwo/news-classifier/LICENSE) file for details.

---

## Contact

For questions or feedback, please contact:

- **Your Name**: [codebasetwo](mailto:codebasetwo@gmail.com)
- **Project Repository**: [GitHub Repository](https://github.com/codebasetwo/news-classifier)

---

Thank you for using the Newsfeed Classification Project! We hope this documentation helps you understand and utilize the project effectively.