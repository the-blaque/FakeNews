import data_preprocessing
import model_selection
import model_evaluation

def main():
    # Step 1: Preprocess the data
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess_data()

    # Step 2: Train the models and select the best one
    best_model = model_selection.select_model(X_train, y_train, X_test, y_test)

    # Step 3: Evaluate the selected model
    model_evaluation.evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
