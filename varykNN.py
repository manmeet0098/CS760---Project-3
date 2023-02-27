import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# read data
data = pd.read_csv('emails.csv')

# split into X and y
X = data.iloc[:,1:-1]
y = data['Prediction']

values_acc = 0
avg_acc = []
for k in [1, 3, 5, 7, 10]:
    # print("Value of k is : " + str(k))

    # initialize kNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

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
        values_acc = acc + values_acc
    avg_acc.append(values_acc / 5)

plt.plot([1, 3, 5, 7, 10], avg_acc, color = 'blue')
plt.grid()
plt.xlabel("k values")
plt.ylabel("Accuracy Values")
plt.show()
