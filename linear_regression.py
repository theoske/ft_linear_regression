import matplotlib.pyplot as plt
import pandas as pd

print("Welcome to ft_linear_regression!\nEnter your car mileage: ")
input_value = float(input())

theta_df = pd.read_csv('theta.csv')
theta0 = theta_df['theta0'].loc[theta_df.index[0]]
theta1 = theta_df['theta1'].loc[theta_df.index[0]]

def ft_linear_regression(theta0, theta1, input_value):
    return theta0 + (theta1 * input_value)

result = ft_linear_regression(theta0, theta1, input_value)
print (f"The estimated price of the car is {result}")

data_df = pd.read_csv('data.csv')

x_values = data_df["km"]
y_values = theta0 + theta1 * x_values

plt.scatter(data_df["km"], data_df["price"], label="Actual Data", color="blue")

plt.plot(x_values, y_values, color="red", label="Regression Line")

plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("ft_linear_regression")
plt.legend()
plt.show()