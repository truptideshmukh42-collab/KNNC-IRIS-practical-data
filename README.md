# 🌸 KNN Classifier – Iris Dataset Practical

## 📌 Aim

To implement the **K-Nearest Neighbors (KNN) Classification algorithm** on the **Iris dataset** and predict the species of Iris flowers.

---

## 📖 Introduction

**K-Nearest Neighbors (KNN)** is a supervised machine learning algorithm used for classification and regression problems.

In this practical, KNN is used to classify Iris flowers into three different species based on their flower measurements.

### Iris Species

* 🌷 Iris Setosa
* 🌷 Iris Versicolor
* 🌷 Iris Virginica

---

## 📊 Dataset

The **Iris dataset** contains **150 observations** with 4 input features:

| Feature      | Description         |
| ------------ | ------------------- |
| Sepal Length | Length of the sepal |
| Sepal Width  | Width of the sepal  |
| Petal Length | Length of the petal |
| Petal Width  | Width of the petal  |

The target variable is the **Iris species**.

---

## 🛠️ Technologies Used

* Python 🐍
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## ⚙️ Algorithm

### Steps:

1. Import the required libraries.
2. Load the Iris dataset.
3. Separate the features and target variable.
4. Split the dataset into training and testing sets.
5. Create the KNN classifier.
6. Train the model using the training dataset.
7. Predict the classes for the test dataset.
8. Calculate the model accuracy.
9. Display the classification results.

---

## 💻 Python Code

```python
# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

# Display dataset
print(df.head())

# Separate features and target
X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

## 📈 Expected Output

The KNN model should provide a high classification accuracy on the Iris dataset.

Example:

```text
Accuracy: 1.0

Classification Report:

              precision    recall    f1-score

Iris Setosa      1.00       1.00       1.00
Iris Versicolor  1.00       1.00       1.00
Iris Virginica   1.00       1.00       1.00
```

*The exact accuracy may vary depending on the train-test split and K value.*

---

## 🧠 Understanding K in KNN

The value of **K** represents the number of nearest neighbors considered when making a prediction.

For example:

```python
KNeighborsClassifier(n_neighbors=5)
```

means the algorithm considers the **5 nearest data points** to classify a new observation.

Different values of K can be tested to find the best-performing model.

---

## 📌 Advantages of KNN

* Simple and easy to understand.
* Easy to implement.
* No complex mathematical assumptions.
* Works well on small datasets.
* Useful for classification problems.

---

## ⚠️ Limitations of KNN

* Can be slow for large datasets.
* Sensitive to the choice of K.
* Sensitive to feature scaling.
* Requires storing the training data.

---

## 🎯 Conclusion

The **K-Nearest Neighbors (KNN)** algorithm was successfully implemented on the **Iris dataset**. The model can classify Iris flowers into **Setosa, Versicolor, and Virginica** based on their sepal and petal measurements.

---

## 📁 Project Structure

```text
KNN-Iris-Practical/
│
├── knnc_iris.py
├── README.md
└── requirements.txt
```

---

## 📦 Requirements

Create a `requirements.txt` file:

```text
numpy
pandas
matplotlib
scikit-learn
```

Install the required libraries using:

```bash
pip install -r requirements.txt
```

---

## 👩‍💻 Author

**Trupti Priyadarshi**

⭐ If you found this practical useful, consider giving the repository a star!
