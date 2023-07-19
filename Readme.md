# Fake News Detection AI

This project aims to build an artificial intelligence model to detect fake news articles. We employ Natural Language Processing (NLP) and various machine learning algorithms to classify articles as either 'True' or 'Fake'.

## Project Structure

The project has the following structure:
- `data_preprocessing.py`: Preprocesses the data, including cleaning, tokenization, and vectorization.
- `model_selection.py`: Trains various models on the data and selects the most accurate one.
- `model_evaluation.py`: Evaluates the performance of the selected model and generates a confusion matrix.
- `main.py`: Master script to call the other scripts and run the entire process.
- `requirements.txt`: Lists the Python dependencies required for this project.
- `install_dependencies.py`: alternative python script to install dependencies

## Prerequisites

Before running this project, make sure to have Python installed on your system. The project has been tested using Python 3.8, but should work with other Python 3 versions as well.

## Setup

1. Clone this repository or download the project's zip to your local machine.

2. Install the necessary libraries and dependencies using the following command:

    ```
    pip install -r requirements.txt
    ```

    or alternatively run the install_dependencies.py script

3. Download the "Fake and real news" dataset from Kaggle (if not already included in project resources) and place the "True.csv" and "Fake.csv" files in the same directory as the Python scripts.

## Usage

To run the entire process, simply execute the `main.py` script:


This will preprocess the data, train the models, select the best model, and evaluate its performance.

## Output

After running the scripts, you should see printouts of the accuracy for each trained model and the accuracy, precision, recall, and F1 score for the chosen model. A confusion matrix for the chosen model will be displayed as a heatmap.

## Libraries Used

- pandas: Data manipulation and analysis.
- nltk: Natural language processing.
- scikit-learn: Machine learning.
- seaborn: Statistical data visualization.
- matplotlib: Plotting library.

## Contributors

Oluwaseeni Ajayi

## License

This project is licensed under the terms of the MIT license.
