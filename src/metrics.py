from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate(y_true, y_pred):
    """
    This is a utility function to evaluate the performance of a model using three metrics: 
    1) Mean Absolute Error (MAE)
    2) Root Mean Squared Error (RMSE)
    3) Weighted Absolute Percentage Error (WAPE)

    Args:
        y_true (array-like): The true values of the target variable.
        y_pred (array-like): The predicted values from the model.

    Returns:
        mae (float): Mean Absolute Error of the predictions.
        rmse (float): Root Mean Squared Error of the predictions.
        wape (float): Weighted Absolute Percentage Error of the predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()   # 加权绝对百分比误  # 加权绝对百分比误差，对 0 销量稳健
    return mae, rmse, wape