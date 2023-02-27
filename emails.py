import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold

# read data
data = pd.read_csv('emails.csv')

# split into X and y
X = data.iloc[:,1:-1]
y = data['Prediction']

# single train/test split
X_train = X.iloc[:4000, :]
X_test = X.iloc[4000:, :]
y_train = y.iloc[:4000]
y_test = y.iloc[4000:]


# initialize 1NN classifier
knn = KNeighborsClassifier(n_neighbors=1)

# perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # train the model
    knn.fit(X_train, y_train)

    # make predictions on test set
    y_pred = knn.predict(X_test)

    # calculate accuracy, precision, and recall
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # print results for each fold
    print(f"Fold {i+1}:")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print()
