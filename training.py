import numpy as np
import pandas as pd

data_df = pd.read_csv('data.csv')

km_mean = data_df["km"].mean()
km_std = data_df["km"].std()
price_mean = data_df["price"].mean()
price_std = data_df["price"].std()

#data standardisation to avoid risk of overflow
data_df["km"] = (data_df["km"] - km_mean) / km_std
data_df["price"] = (data_df["price"] - price_std) / price_std


m = len(data_df.index)
theta_df = pd.DataFrame(['theta0', 'theta1'])

def get_theta():
    learningRate = 0.0001
    theta0 = 0
    theta1 = 0
    previous_cost = None
    for iteration in range(100000000):
        predictions = theta0 + (theta1 * data_df["km"])
        current_cost = mean_squared_error(np.array(data_df['price']), np.array(predictions))
        if previous_cost is not None and abs(previous_cost-current_cost)<=1e-6:
            break
        previous_cost = current_cost
        tmp0 = learningRate * (predictions.sum() - data_df["price"].sum()) / m
        tmp1 = learningRate * get_sum_of_diff(predictions, data_df['price'], data_df['km']) / m
        theta0 -= tmp0
        theta1 -= tmp1
    return theta0, theta1

def mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost

def get_sum_of_diff(predictions, real_price, kilometer):
    if len(predictions) != len(real_price):
        raise ValueError("DataFrame column and list must have the same length")
    p_array = np.array(predictions)
    r_array = np.array(real_price)
    k_array = np.array(kilometer)
    diff = np.sum((p_array - r_array) * k_array)
    return diff

theta0, theta1 = get_theta()

#data destandardisation to put it back to scale
theta1 = theta1 * (price_std / km_std)
theta0 = price_mean - (theta1 * km_mean)

res_df = pd.DataFrame({"theta0":[theta0], "theta1":[theta1]})

res_df.to_csv("theta.csv", index=False)