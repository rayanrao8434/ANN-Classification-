# Predictive Diabetes Intelligence 🧬

A Deep Learning web application that predicts the likelihood of diabetes based on patient vitals. 

This project uses an **Artificial Neural Network (ANN)** built with TensorFlow/Keras, trained on the Pima Indians Diabetes dataset, and deployed using a modern, dark-themed **Streamlit** user interface.

## Features
- **Deep Learning Model**: A fully connected Artificial Neural Network optimized for binary classification.
- **Sleek UI/UX**: A dark-themed, responsive, and modern web interface built with Streamlit and custom CSS.
- **Instant Predictions**: Real-time inference scaling patient vitals (Glucose, BMI, Age, etc.) using `joblib` and processing them through the neural network.

## Project Structure
- `app.py`: The Streamlit web application script.
- `ann_diabetes.py`: The script used for data preprocessing, building, training, and saving the improved ANN model.
- `diabetes_model.h5`: The trained Keras HDF5 model.
- `scaler.joblib`: The saved StandardScaler object used to normalize input features to match the training data distribution.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rayanrao8434/ANN-Classification-.git
   cd ANN-Classification-
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   Make sure you have the following packages installed:
   ```bash
   pip install streamlit numpy pandas scikit-learn tensorflow joblib
   ```

## Usage

**Running the Web App:**
To start the Streamlit application locally, run the following command in your terminal:
```bash
streamlit run app.py
```
This will open the web interface in your default browser. Enter the patient's medical metrics (like Age, BMI, Blood Pressure, etc.) and click **Predict** to see the probability of diabetes risk.

**Retraining the Model:**
If you wish to experiment with the neural network architecture or retrain the model, run:
```bash
python ann_diabetes.py
```
This will process the data, train the model, evaluate the accuracy, and overwrite `diabetes_model.h5` and `scaler.joblib` with the new versions.

## Technologies Used
- **Python 3**
- **TensorFlow / Keras** (Deep Learning)
- **Scikit-Learn** (Data Preprocessing / Scaling)
- **Pandas & NumPy** (Data Manipulation)
- **Streamlit** (Frontend Web Framework)

## License
This project is open-source and available under the [MIT License](LICENSE).
