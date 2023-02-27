import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Load training set from D2z.txt
train_data = np.loadtxt('D2z.txt', delimiter=' ', usecols=(0,1))
train_labels = np.loadtxt('D2z.txt', delimiter=' ', usecols=2)

# Generate test grid
x_min, x_max = -2, 2
y_min, y_max = -2, 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
test_data = np.c_[xx.ravel(), yy.ravel()]

# Train 1NN classifier
classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
classifier.fit(train_data, train_labels)

# Predict on test grid
predictions = classifier.predict(test_data)

# Plot training set and test predictions
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap='coolwarm')
plt.scatter(test_data[:, 0], test_data[:, 1], c=predictions, cmap='coolwarm', alpha=0.2)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('1NN Predictions on 2D Grid')
plt.show()
