import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main():
    bikers = pd.read_csv('biker_data.csv')
    print("part 1")

    #data modifications
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
    for bridge in bridges:
        for i in range(len(bikers[bridge])):
            bikers.at[i, bridge] = int(bikers.at[i, bridge].replace(',', ''))
    for i in range(len(bikers['Total'])):
        bikers.at[i, 'Total'] = int(bikers.at[i, 'Total'].replace(',', ''))

    print("part 2")
    snow = []
    for i in range(len(bikers['Precipitation'])):
        if "(S)" in str(bikers.at[i, 'Precipitation']):
            snow.append(1)
            bikers.at[i, 'Precipitation'] = float(bikers.at[i, 'Precipitation'].replace('(S)', ''))
        else:
            snow.append(0)
    bikers["Snow"] = snow

    for i in range(len(bikers['Precipitation'])):
        if "T" in str(bikers.at[i, 'Precipitation']):
            bikers.at[i, 'Precipitation'] = 0.01
        else:
            bikers.at[i, 'Precipitation'] = float(bikers.at[i, 'Precipitation'])

    #predict_bridge(bikers, bridges)

    #predict_weather(bikers, bridges)

    predict_precip(bikers, bridges)

def predict_precip(bikers, bridges):
    print("part 4")
    X = bikers[['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']]
    y = bikers[['Precipitation']]

    print("part 5")
    X = X.to_numpy()
    y = y.to_numpy()

    print("part 6")
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)
    lmbda = np.logspace(-1, 2, num=200)

    MODEL = []
    MSE = []

    print("part 7")
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    ind =  MSE.index(min(MSE))
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]
    print(model_best)

    plt.plot(range(len(X_test)), pow(y_test - model_best.predict(X_test), 2), label="Squared error of Precipitation Based on Bike Traffic: " + str(int(MSE_best)))
    plt.legend(loc="upper left")
    plt.title("r^2 = " + str(model_best.score(X_test, y_test)))
    plt.ylabel("Squared Error")
    plt.xlabel("Date")
    plt.show()

def predict_weather(bikers, briges):
    print("part 4")
    X = bikers[['High Temp (°F)', 'Low Temp (°F)', 'Precipitation', 'Snow']]
    y = bikers[['Total']]

    print("part 5")
    X = X.to_numpy()
    y = y.to_numpy()

    print("part 6")
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)
    lmbda = np.logspace(-1, 2, num=200)

    MODEL = []
    MSE = []

    print("part 7")
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    ind =  MSE.index(min(MSE))
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]
    print(model_best)

    plt.plot(range(len(X_test)), pow(y_test - model_best.predict(X_test), 2), label="Squared error of total cyclists due to weather: " + str(int(MSE_best)))
    plt.legend(loc="upper left")
    plt.title("Squared Error of the Predicted Number of Cyclists on a Bridge Based on the Weather; r^2 = " + str(model_best.score(X_test, y_test)))
    plt.ylabel("Squared Error")
    plt.xlabel("Date")
    plt.show()

def predict_bridge(bikers, bridges):
    models = {}

    print("part 3")
    for i in range(4):
        print("part 4")
        X = bikers[[bridges[(0 + i) % 4], bridges[(1 + i) % 4], bridges[(2 + i) % 4]]]
        y = bikers[[bridges[(3 + i) % 4]]]

        print("part 5")
        X = X.to_numpy()
        y = y.to_numpy()

        print("part 6")
        [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)
        [X_train, trn_mean, trn_std] = normalize_train(X_train)
        X_test = normalize_test(X_test, trn_mean, trn_std)
        lmbda = np.logspace(-1, 2, num=51)

        MODEL = []
        MSE = []

        print("part 7")
        for l in lmbda:
            #Train the regression model using a regularization parameter of l
            model = train_model(X_train,y_train,l)

            #Evaluate the MSE on the test set
            mse = error(X_test,y_test,model)

            #Store the model and mse in lists for further processing
            MODEL.append(model)
            MSE.append(mse)

        ind =  MSE.index(min(MSE))
        [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

        models[bridges[(3 + i) % 4]] = {
            "lambda": lmda_best,
            "mse": MSE_best,
            "model": model_best,
        }

        plt.plot(range(len(X_test)), pow(y_test - model_best.predict(X_test), 2), label=str(bridges[(3 + i) % 4]) + ": " + str(model_best.score(X_test, y_test)))

    print("Does this work")
    print(models)
    plt.legend(loc="upper left")
    plt.title("Squared Error of the Predicted Number of Cyclists on a Bridge Based on the Traffic on Other Bridges")
    plt.ylabel("Squared Error")
    plt.xlabel("Date")
    plt.show()

def normalize_train(X_train):
    print(X_train)
    cols = len(X_train[0])
    print(len(X_train[0]))
    print(X_train[:, 0])
    print(np.mean(X_train[:, 0]))
    trn_mean = [np.mean(X_train[:, col]) for col in range(cols)]
    trn_std = [np.std(X_train[:, col]) for col in range(cols)]
    X = [[(x - trn_mean[col]) / trn_std[col] for col, x in enumerate(row)] for row in X_train]

    return X, trn_mean, trn_std

def normalize_test(X_test, trn_mean, trn_std):
    return [[(x - trn_mean[col]) / trn_std[col] for col, x in enumerate(row)] for row in X_test]

def train_model(X,y,l):
    return Ridge(alpha=1).fit(X, y)

def error(X,y,model):
    return mean_squared_error(y, model.predict(X))

if __name__ == '__main__':
    main()
