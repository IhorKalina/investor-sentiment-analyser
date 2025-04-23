import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'crude_oil_reddit_posts_updated.csv'
data = pd.read_csv(file_path)

X = data[['Sentiment']]
y = data['Price oil index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

rf_reg = RandomForestRegressor(random_state=42, n_estimators=100)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
y_pred_knn = knn_reg.predict(X_test)

def evaluate_model(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} - MSE: {mse:.2f}, RMSE: {rmse:.2f}, R^2: {r2:.2f}')

evaluate_model("Linear Regression", y_test, y_pred_lin)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("K-Nearest Neighbors", y_test, y_pred_knn)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

axes[0].scatter(X_test, y_test, color='blue', label='Actual data', alpha=0.7)
axes[0].scatter(X_test, y_pred_lin, color='red', label='Predictions', alpha=0.7)
axes[0].set_title('Linear Regression')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Price oil index')
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(X_test, y_test, color='blue', label='Actual data', alpha=0.7)
axes[1].scatter(X_test, y_pred_rf, color='green', label='Predictions', alpha=0.7)
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Sentiment')
axes[1].legend()
axes[1].grid(True)

axes[2].scatter(X_test, y_test, color='blue', label='Actual data', alpha=0.7)
axes[2].scatter(X_test, y_pred_knn, color='orange', label='Predictions', alpha=0.7)
axes[2].set_title('K-Nearest Neighbors')
axes[2].set_xlabel('Sentiment')
axes[2].legend()
axes[2].grid(True)

fig.suptitle('Comparison of Model Predictions', fontsize=16)
plt.tight_layout()
plt.show()
