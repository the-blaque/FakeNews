from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    print(f'{model.__class__.__name__} accuracy:', model.score(X_test, y_test))
    return model

def select_model(X_train, y_train, X_test, y_test):
    # train and evaluate a logistic regression model
    lr = LogisticRegression(max_iter=2000)
    model_lr = train_and_evaluate_model(lr, X_train, y_train, X_test, y_test)

    # train and evaluate a Naive Bayes model
    nb = MultinomialNB()
    model_nb = train_and_evaluate_model(nb, X_train, y_train, X_test, y_test)

    # train and evaluate a SVM model
    clf = svm.SVC()
    model_svm = train_and_evaluate_model(clf, X_train, y_train, X_test, y_test)

    # Placeholder for best model. You might want to implement a way to compare the models and return the best one.
    best_model = model_lr
    return best_model
