# Yangtze River Refined Oil Freight Rate Prediction

This repository contains the code and data used for predicting the freight rates of refined oil on the Yangtze River. The project involves various machine learning and statistical models to analyze and forecast the prices.

## Project Structure

- **Data**
  - `price_data.xlsx`: Contains the historical data of refined oil freight rates.

- **Scripts**
  - `ARIMA-RF.py`: Implementation of the ARIMA and Random Forest hybrid model.
  - `correlation analysis.py`: Performs correlation analysis between different variables.
  - `EMD-ARIMA.py`: Empirical Mode Decomposition (EMD) and ARIMA combined model.
  - `Kalman-BP.py`: Combines Kalman filter with Back Propagation neural network.
  - `LSTM-ARIMA.py`: Hybrid model using Long Short-Term Memory (LSTM) networks and ARIMA.
  - `Ridge regression.py`: Implements ridge regression for price prediction.

## Models

### 1. ARIMA and Random Forest (ARIMA-RF)
This model combines the strengths of both ARIMA for capturing linear patterns and Random Forest for capturing non-linear relationships.

### 2. Correlation Analysis
This script analyzes the correlation between different features in the dataset to identify significant predictors.

### 3. EMD and ARIMA (EMD-ARIMA)
EMD is used to decompose the time series data into Intrinsic Mode Functions (IMFs), and ARIMA is applied to these IMFs for better prediction accuracy.

### 4. Kalman Filter and Back Propagation (Kalman-BP)
The Kalman filter is used for noise reduction in the data, and the BP neural network is applied for prediction.

### 5. LSTM and ARIMA (LSTM-ARIMA)
This hybrid model leverages LSTM networks to capture long-term dependencies in the data, while ARIMA is used for short-term forecasting.

### 6. Ridge Regression
Ridge regression is applied to handle multicollinearity and improve the robustness of the prediction model.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yangtze-freight-rate-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd yangtze-freight-rate-prediction
   ```
3. Ensure you have all required dependencies installed. You can use `requirements.txt` if provided:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the desired script:
   ```bash
   python ARIMA-RF.py
   ```

## Data

The data used for this project is included in the `price_data.xlsx` file. This contains the historical freight rates which are crucial for training and evaluating the models.

## Results

The performance of each model is evaluated based on metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. Detailed results and comparisons can be found within the respective scripts.

## Contributing

If you wish to contribute to this project, please fork the repository and create a pull request with your changes. Ensure that your contributions are well-documented and tested.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any inquiries or questions, please contact [yourname@domain.com](mailto:yourname@domain.com).

```

Feel free to modify the details such as the repository URL, contact email, and any additional instructions or descriptions that are specific to your project.
