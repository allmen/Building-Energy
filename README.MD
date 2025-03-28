# Energy Consumption Predictive Analysis

This repository contains a Jupyter Notebook for predicting energy consumption using a variety of machine learning techniques. The data used in this notebook comes from `energy.csv`, which includes the following features:

- `TOTAL_FLOOR_AREA`: The total floor area of the building.
- `NUMBER_HEATED_ROOMS`: The number of heated rooms in the building.
- `CURRENT_ENERGY_EFFICIENCY`: The current energy efficiency rating of the building.
- `energy`: The energy consumption of the building.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Visualization](#visualization)

## Installation

To run the notebook, you need to have Python 3.9 and the following packages installed:

```bash
pip install tensorflow pandas matplotlib seaborn scikit-learn keras
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/allmen/Building-Energy.git
    cd Building-Energy
    ```

2. Open the Jupyter Notebook:
    ```bash
    jupyter notebook energy.ipynb
    ```

## Data Preprocessing

The notebook performs the following preprocessing steps:

1. Load the data from `energy.csv`:
    ```python
    df = pd.read_csv('energy.csv')
    ```

2. Display the first few rows of the dataset:
    ```python
    df.head()
    ```

3. Check for missing values and handle them:
    ```python
    data = df.dropna()
    ```

4. Normalize the data using `MinMaxScaler`:
    ```python
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    ```

## Model Training

The notebook includes code for training different models, including Linear Regression, Polynomial Regression, and RNN-LSTM. Below is an example of the RNN-LSTM model:

1. Prepare the dataset:
    ```python
    xtrain, ytrain = prepare_dataset(scaled_data[:train_size], 30)
    xval, yval = prepare_dataset(scaled_data[train_size-2:train_size +test_size], 30)
    xtest, ytest = prepare_dataset(scaled_data[train_size + test_size-2:],30)
    ```

2. Define and compile the RNN-LSTM model:
    ```python
    model_RNN_LSTM = Sequential()
    model_RNN_LSTM.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], X_train.shape[2])))
    model_RNN_LSTM.add(SimpleRNN(64, return_sequences=True))
    model_RNN_LSTM.add(LSTM(64, return_sequences = True, activation = 'relu'))
    model_RNN_LSTM.add(Flatten())
    model_RNN_LSTM.add(Dropout(0.2))
    model_RNN_LSTM.add(Dense(1))
    model_RNN_LSTM.compile(loss = 'mse', metrics = metric)
    ```

3. Train the model:
    ```python
    history=model_RNN_LSTM.fit(X_train, y_train, epochs=100,  validation_data=(X_val, y_val), verbose=0, batch_size=64, callbacks=early_stopping)
    ```

## Evaluation

The notebook includes methods for evaluating the model's performance using metrics such as RMSE, MSE, MAE, and R^2 Score. Example:
```python
RNN_LSTM_Predict = model_RNN_LSTM.predict(X_test)
predictions_RNN = scaler.inverse_transform(RNN_LSTM_Predict)
y_test_inverse= scaler.inverse_transform(y_test)

RNN_rmse = np.sqrt(mean_squared_error(predictions_RNN, y_test_inverse))
RNN_mae= (mean_absolute_error(predictions_RNN, y_test_inverse))
RNN_mse= (mean_squared_error(predictions_RNN, y_test_inverse))
RNN_r1= (r2_score(predictions_RNN, y_test_inverse))
```

## Visualization

The notebook provides various visualizations to understand the data and model performance:
1. Heatmap of the correlations in the dataset:
    ```python
    fig= plt.subplots(figsize=(16,12))
    corr = data.corr()
    sns.heatmap(corr, cmap = 'Wistia', annot = True)
    plt.show()
    ```

2. Histogram of the `energy` feature:
    ```python
    plt.hist(data["energy"])
    plt.xlabel("energy")
    plt.ylabel('Frequency')
    plt.show()
    ```

3. Plot of actual vs predicted results:
    ```python
    plt.plot(predictions_RNN[:100], '--', label='Predict')
    plt.plot(y_test_inverse[:100], '--', label='Actual')
    plt.title('Actual vs Predicted Results')
    plt.legend()
    ```

## Conclusion

This notebook demonstrates a comprehensive approach to predicting building energy consumption using various machine learning models, including preprocessing, model training, evaluation, and visualization.
