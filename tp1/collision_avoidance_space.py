import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def linear_regression(X_train, X_val, y_train, y_val):
    reg = LinearRegression().fit(X_train, y_train)

    y_train_pred = reg.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)

    y_val_pred = reg.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)

    return train_mse, val_mse, y_train_pred, y_val_pred, reg


def model_selection(X_train, y_train, X_val, y_val):
    train_mses = []
    val_mses = []
    y_train_preds = []
    y_val_preds = []
    models = []
    for degree in range(1, 7):
        poly = PolynomialFeatures(degree)
        X_r = poly.fit_transform(X_train)
        X_v = poly.fit_transform(X_val)

        train_mse, val_mse, y_train_pred, y_val_pred, model = linear_regression(X_r, X_v, y_train, y_val)
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        y_train_preds.append(y_train_pred)
        y_val_preds.append(y_val_pred)
        models.append(model)

    return models, train_mses, val_mses, y_train_preds, y_val_preds


def plot_mse(train_mses, val_mses):
    plt.yscale('log')
    dim = np.arange(1, 7)
    plt.plot(dim, train_mses, '-x', label='train')
    plt.plot(dim, val_mses, '-o', label='val')
    plt.xlabel('degree')
    plt.ylabel('mse')

    plt.legend()
    plt.show()


def plot_true_vs_predicted(y_train, y_val, y_train_preds, y_val_preds, best_model_degree):
    fig = plt.figure(figsize=(12, 8))

    for degree in range(1, 7):
        plt.subplot(230+degree)

        plt.plot(y_train.flatten(), y_train_preds[degree-1], '.b', label='train')
        plt.plot(y_val.flatten(), y_val_preds[degree-1], '.r', label='val')

        plt.axline((0, 0), slope=1, color='g')

        plt.legend()
        plt.title(f'Degree {degree}' if degree != best_model_degree else f'Best = Degree {degree}')

    fig.text(0.5, 0.009, 'True Miss Distance Values', ha='center', va='center', size=12)
    fig.text(0.009, 0.5, 'Predicted Miss Distance Values', ha='center', va='center', rotation='vertical', size=12)
    plt.tight_layout()

    plt.show()


def collision_avoidance_space(path='../data/SatelliteConjunctionDataRegression.csv'):
    dataset = pd.read_csv(path)

    X = dataset[dataset.columns.values.tolist()[:len(dataset.columns)-1]]

    y = dataset[dataset.columns.values.tolist()[-1]]

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.20)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)

    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()

    y_train = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))
    y_val = y_scaler.transform(y_val.to_numpy().reshape(-1, 1))
    y_test = y_scaler.transform(y_test.to_numpy().reshape(-1, 1))

    models, train_mses, val_mses, y_train_preds, y_val_preds = model_selection(X_train, y_train, X_val, y_val)

    best_model_index = np.argmin(val_mses, axis=0)
    best_model_degree = best_model_index+1

    plot_mse(train_mses, val_mses)
    plot_true_vs_predicted(y_train, y_val, y_train_preds, y_val_preds, best_model_degree)

    best_model = models[best_model_index]
    poly = PolynomialFeatures(best_model_degree)
    X_t = poly.fit_transform(X_test)

    y_test_pred = best_model.predict(X_t)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f'Best model = Degree {best_model_degree}\nTest error = {test_mse}')
