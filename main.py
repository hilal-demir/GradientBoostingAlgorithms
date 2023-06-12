import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import catboost as ctb
import lightgbm as lgb
import matplotlib.pyplot as plt
import sklearn.metrics

from xgboost import plot_tree
from lightgbm import plot_tree as lgbm_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier


# Load the data from the Excel file
data = pd.read_excel('/Users/ibm/BOUN/datacopy.xlsx')

# Define the target column and convert it to numerical labels
target_map = {"+": 1, "-": 0, "n": 2}
data["label"] = data["label"].map(target_map)

# Define the feature and target data
features = data["ileti"]
target = data["label"]


# Split the data into training and testing sets
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

# Transform documents to document-term matrix
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_features)
test_features = vectorizer.transform(test_features)

# Confusion matrix for multiclass
def conf_matrix(predicted_y):
    cm = confusion_matrix(test_target, predicted_y)
    cm_df = pd.DataFrame(cm,
                         index=['negative', 'positive', 'neutral'],
                         columns=['negative', 'positive', 'neutral'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()

# XGBoost Model
def xgb_boost():
    global train_features, test_features

    # Define the XGBoost model and train it on the training data
    dtrain = xgb.DMatrix(train_features, label=train_target)
    params = {"objective": "multi:softmax", "num_class": 3, "num_leaves": 3, "learning_rate": 0.05}
    model = xgb.train(params, dtrain)

    # Predict the labels for the test data and calculate the accuracy
    dtest = xgb.DMatrix(test_features)
    preds = model.predict(dtest)

    # Calculate accuracy score and f1 score
    accuracy = accuracy_score(test_target, preds)
    print("Accuracy: %.2f%%" % (accuracy))
    f1 = sklearn.metrics.f1_score(test_target, preds, average='micro')
    print("f1:", f1)

    # decision tree
    plot_tree(model, num_trees=3, rankdir='LR')
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.show()

    # Confusion matrix
    #conf_matrix(preds)

# XGBoost Model with cross-validation
def xgb_boost_k_fold():
    global features, target

    params = {"objective": "multi:softmax", "num_class": 3, "num_leaves": 3, "learning_rate": 0.05}

    # Perform 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    a_scores = []
    f_scores = []
    for train_idx, test_idx in cv.split(features, target):
        train_features, test_features = features[train_idx], features[test_idx]
        train_target, test_target = target[train_idx], target[test_idx]

        # Transform documents to document-term matrix
        vectorizer = TfidfVectorizer()
        train_features = vectorizer.fit_transform(train_features)
        test_features = vectorizer.transform(test_features)


        dtrain = xgb.DMatrix(train_features, label=train_target)

        model = xgb.train(params, dtrain)

        # Predict the labels for the test data and calculate the accuracy
        dtest = xgb.DMatrix(test_features)
        preds = model.predict(dtest)

        # Calculate accuracy score and f1 score
        accuracy = accuracy_score(test_target, preds)
        print("Accuracy: %.2f%%" % (accuracy))
        a_scores.append(accuracy)
        f1 = sklearn.metrics.f1_score(test_target, preds, average='micro')
        print("f1:", f1)
        f_scores.append(f1)

    # Print the cross-validation scores
    a_scores = np.array(a_scores)
    print("Accuracy Cross-validation scores:", a_scores)
    print("Average accuracy: %.2f%%" % (a_scores.mean() * 100))
    print("Standard deviation: %.2f%%" % (a_scores.std() * 100))

    # Print the cross-validation scores
    f_scores = np.array(f_scores)
    print("f1 Cross-validation scores:", f_scores)
    print("Average accuracy: %.2f%%" % (f_scores.mean() * 100))
    print("Standard deviation: %.2f%%" % (f_scores.std() * 100))

# LightGBM Model
def lightgbm():
    global train_features, test_features

    # Define the training dataset
    train_data = lgb.Dataset(train_features, label=train_target)

    # Set the hyperparameters for the LightGBM model
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_classes": 3,
        "num_leaves": 3,
        "learning_rate": 0.05
    }

    # Train the LightGBM model
    model = lgb.train(params, train_data, num_boost_round=100)

    # Use the model to make predictions on the testing data
    y_pred = model.predict(test_features)


    # Convert predictions from one-hot encoding to class indices
    y_pred = np.argmax(y_pred, axis=1)

    # Calculate accuracy score and f1 score
    accuracy = accuracy_score(test_target, y_pred)
    f1 = sklearn.metrics.f1_score(test_target, y_pred, average='micro')
    print("Accuracy:", accuracy)
    print("f1:", f1)

    # decision tree
    lgbm_tree(model, tree_index=4)
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.show()

    # Confusion matrix
    conf_matrix(y_pred)

# LightGBM Model with cross-validation
def lightgbm_k_fold():
    global features, target

    # Set the hyperparameters for the LightGBM model
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_classes": 3,
        "num_leaves": 3,
        "learning_rate": 0.05
    }

    # Perform 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    a_scores = []
    f_scores = []
    for train_idx, test_idx in cv.split(features, target):
        train_features, test_features = features[train_idx], features[test_idx]
        train_target, test_target = target[train_idx], target[test_idx]

        # Transform documents to document-term matrix
        vectorizer = TfidfVectorizer()
        train_features = vectorizer.fit_transform(train_features)
        test_features = vectorizer.transform(test_features)

        # Define the training dataset
        train_data = lgb.Dataset(train_features, label=train_target)

        # Train the LightGBM model
        model = lgb.train(params, train_data, num_boost_round=100)

        # Use the model to make predictions on the testing data
        preds = model.predict(test_features)

        # Convert predictions from one-hot encoding to class indices
        preds = np.argmax(preds, axis=1)

        # Calculate accuracy score and f1 score
        accuracy = accuracy_score(test_target, preds)
        print("Accuracy: %.2f%%" % (accuracy))
        a_scores.append(accuracy)
        f1 = sklearn.metrics.f1_score(test_target, preds, average='micro')
        print("f1:", f1)
        f_scores.append(f1)

    # Print the cross-validation scores
    a_scores = np.array(a_scores)
    print("Accuracy Cross-validation scores:", a_scores)
    print("Average accuracy: %.2f%%" % (a_scores.mean() * 100))
    print("Standard deviation: %.2f%%" % (a_scores.std() * 100))

    # Print the cross-validation scores
    f_scores = np.array(f_scores)
    print("f1 Cross-validation scores:", f_scores)
    print("Average accuracy: %.2f%%" % (f_scores.mean() * 100))
    print("Standard deviation: %.2f%%" % (f_scores.std() * 100))



# CatBoost Model
def catboost():
    global train_features, test_features


    # Define the CatBoost model and train it on the training data
    pool = ctb.Pool(train_features, train_target)
    cf = ctb.CatBoostClassifier(iterations=10, learning_rate=0.05,depth=3,loss_function='MultiClass', verbose=False)
    model_CBC = cf.fit(pool)

    # decision tree
    decision_tree = model_CBC.plot_tree(tree_idx=3)
    decision_tree.render()

    # Use the model to make predictions on the testing data
    expected_y = test_target
    predicted_y = model_CBC.predict(test_features)

    # Calculate accuracy score and f1 score
    accuracy = accuracy_score(expected_y, predicted_y)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    f1 = sklearn.metrics.f1_score(test_target, predicted_y, average='micro')
    print("f1:", f1)

    # Confusion matrix
    conf_matrix(predicted_y)

# CatBoost Model with cross-validation
def catboost_k_fold():
    global features, target

    # Perform 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    a_scores = []
    f_scores = []
    for train_idx, test_idx in cv.split(features, target):
        train_features, test_features = features[train_idx], features[test_idx]
        train_target, test_target = target[train_idx], target[test_idx]

        # Transform documents to document-term matrix
        vectorizer = TfidfVectorizer()
        train_features = vectorizer.fit_transform(train_features)
        test_features = vectorizer.transform(test_features)

        # Define the CatBoost model and train it on the training data
        pool = ctb.Pool(train_features, train_target)
        cf = ctb.CatBoostClassifier(iterations=10, learning_rate=0.05, depth=3, loss_function='MultiClass',
                                    verbose=False)
        model_CBC = cf.fit(pool)

        # Use the model to make predictions on the testing data
        preds = model_CBC.predict(test_features)

        # Calculate accuracy score and f1 score
        accuracy = accuracy_score(test_target, preds)
        print("Accuracy: %.2f%%" % (accuracy))
        a_scores.append(accuracy)
        f1 = sklearn.metrics.f1_score(test_target, preds, average='micro')
        print("f1:", f1)
        f_scores.append(f1)

    # Print the cross-validation scores
    a_scores = np.array(a_scores)
    print("Accuracy Cross-validation scores:", a_scores)
    print("Average accuracy: %.2f%%" % (a_scores.mean() * 100))
    print("Standard deviation: %.2f%%" % (a_scores.std() * 100))

    # Print the cross-validation scores
    f_scores = np.array(f_scores)
    print("f1 Cross-validation scores:", f_scores)
    print("Average accuracy: %.2f%%" % (f_scores.mean() * 100))
    print("Standard deviation: %.2f%%" % (f_scores.std() * 100))


# Ensemble Classification Model
def ensemble():
    global train_features, test_features
    # CatBoost Model
    gb1 = ctb.CatBoostClassifier(verbose=False)
    # LightGBM Model
    gb2 = lgb.LGBMClassifier()
    # XGBoost Model
    gb3 = xgb.XGBClassifier()

    # Soft voting classification
    egb = VotingClassifier(estimators=[('cb', gb1), ('lb', gb2), ('xb', gb3)], voting='soft')

    # Use the selected model to make predictions on the testing data
    egb.fit(train_features, train_target)
    y_pred = egb.predict(test_features)

    # Calculate accuracy score and f1 score
    score = accuracy_score(test_target, y_pred)
    print("Accuracy: %.2f%%" % (score * 100.0))
    f1 = sklearn.metrics.f1_score(test_target, y_pred, average='micro')
    print("f1:", f1)

    # Confusion matrix
    conf_matrix(y_pred)

# Ensemble Model with cross-validation
def ensemble_k_fold():
    global features, target

    # CatBoost Model
    gb1 = ctb.CatBoostClassifier(verbose=False)
    # LightGBM Model
    gb2 = lgb.LGBMClassifier()
    # XGBoost Model
    gb3 = xgb.XGBClassifier()

    # Soft voting classification
    egb = VotingClassifier(estimators=[('xb', gb3), ('lb', gb2)], voting='soft')

    # Perform 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    a_scores = []
    f_scores = []
    for train_idx, test_idx in cv.split(features, target):
        train_features, test_features = features[train_idx], features[test_idx]
        train_target, test_target = target[train_idx], target[test_idx]

        # Transform documents to document-term matrix
        vectorizer = TfidfVectorizer()
        train_features = vectorizer.fit_transform(train_features)
        test_features = vectorizer.transform(test_features)

        # Use the selected model to make predictions on the testing data
        egb.fit(train_features, train_target)
        preds = egb.predict(test_features)

        # Calculate accuracy score and f1 score
        accuracy = accuracy_score(test_target, preds)
        print("Accuracy: %.2f%%" % (accuracy))
        a_scores.append(accuracy)
        f1 = sklearn.metrics.f1_score(test_target, preds, average='micro')
        print("f1:", f1)
        f_scores.append(f1)

    # Print the cross-validation scores
    a_scores = np.array(a_scores)
    print("Accuracy Cross-validation scores:", a_scores)
    print("Average accuracy: %.2f%%" % (a_scores.mean() * 100))
    print("Standard deviation: %.2f%%" % (a_scores.std() * 100))

    # Print the cross-validation scores
    f_scores = np.array(f_scores)
    print("f1 Cross-validation scores:", f_scores)
    print("Average accuracy: %.2f%%" % (f_scores.mean() * 100))
    print("Standard deviation: %.2f%%" % (f_scores.std() * 100))

ensemble_k_fold()

# SVM Model
def svm():
    global train_features, test_features

    clf = SVC(kernel='poly', degree=3, gamma=0.05)
    clf.fit(train_features, train_target)
    prediction = clf.predict(test_features)

    # Calculate accuracy score and f1 score
    accuracy = accuracy_score(test_target, prediction)
    print("Accuracy:", accuracy)
    f1 = sklearn.metrics.f1_score(test_target, prediction, average='micro')
    print("f1:", f1)

    # Confusion matrix
    conf_matrix(prediction)

    # RF Model
def rf():
    global train_features, test_features

    classifier = RandomForestClassifier(max_depth = 3, criterion = 'entropy', random_state = 42)

    classifier.fit(train_features, train_target)
    prediction = classifier.predict(test_features)

    # Calculate accuracy score and f1 score
    accuracy = accuracy_score(test_target, prediction)
    print("Accuracy:", accuracy)
    f1 = sklearn.metrics.f1_score(test_target, prediction, average='micro')
    print("f1:", f1)

    # Confusion matrix
    conf_matrix(prediction)

#ensemble()

#catboost()

#lightgbm()

#xgb_boost()

#svm()

#rf()