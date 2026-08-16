# ============================================
# Artificial Neural Network - Diabetes Dataset
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------

# Kaggle URL was unavailable, so use the raw CSV
# from a public mirror of the Pima Indians Diabetes dataset.
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(url, names=columns)

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------
# 2. Basic Preprocessing
# ------------------------------------------------

# In this dataset, some medical measurements use 0
# to represent missing/unrecorded values.
zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace invalid zero values with NaN
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

# Fill missing values using the median of each column
for column in zero_as_missing:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing values after preprocessing:")
print(df.isnull().sum())


# ------------------------------------------------
# 3. Separate Features and Target
# ------------------------------------------------

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ------------------------------------------------
# 4. Split Data into Training and Testing Sets
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------
# 5. Normalize Features
# ------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------------
# 6. Build Artificial Neural Network
# ------------------------------------------------

model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Display model architecture
model.summary()


# ------------------------------------------------
# 7. Compile Model
# ------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------
# 8. Train Model
# ------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ------------------------------------------------
# 9. Evaluate Model on Test Data
# ------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ------------------------------------------------
# 10. Generate Predictions
# ------------------------------------------------

y_probability = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred = (y_probability >= 0.5).astype(int).flatten()

print("\nAccuracy using sklearn:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ------------------------------------------------
# 11. Save Model and Scaler
# ------------------------------------------------

model.save("diabetes_model.h5")
joblib.dump(scaler, "scaler.joblib")
print("\nModel saved as 'diabetes_model.h5'")
print("Scaler saved as 'scaler.joblib'")