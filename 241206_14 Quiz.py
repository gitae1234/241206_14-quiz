import os
import pandas as pd
os.makedirs('./data', exist_ok=True)

results_dir = "./results/"
output_file = os.path.join(results_dir, "correlation_coefficient.csv")
os.makedirs(results_dir, exist_ok=True)

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.DataFrame(columns=column_names)
correlations = data.corr(method='pearson')
correlations.to_csv(output_file)

import matplotlib
import matplotlib.pyplot as pyplot

import numpy as np
import matplotlib.pyplot as plt

filename = "./data/1_pima.csv"
output_file = "./results/correlation_coefficient.csv"

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=column_names)

correlations = data.corr(method='pearson')
print(correlations)
correlations.to_csv("./results/correlation_coefficient.csv")


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(column_names)
ax.set_yticklabels(column_names)

plt.savefig("./results/correlation_plot.png")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

data_path = "./data/08_pima-indians-diabetes.csv"
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)

filename = './data/1_pima.csv.'

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=column_names)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

y_pred_binary = (y_pred > 0.5).astype(int)
print(y_pred_binary)

accuracy = accuracy_score(y_test, y_pred_binary)

print("----------------------------------")
print("Actual Values:", y_test)
print("Predicted Values:", y_pred_binary)
print("----------------------------------")
print("Accuracy:", accuracy)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(y_pred_binary)), y_pred_binary, color='red', label='Predicted Values', marker='x')

plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Data Index')
plt.ylabel('Class (0 or 1)')
plt.legend()

plt.savefig('./linear_regression.png')