
"""models needed to create the Neural Nets"""
from keras.models import Sequential
from keras.layers import Conv1D, Activation, Dropout, Dense, LSTM, Flatten, MaxPooling1D
import tensorflow as tf

"""needed for scaling and calculating the performance of the model"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


"""needed for the decomposition of the time series"""
from PyEMD import CEEMDAN


"""needed for downloading and wrangling data"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import random
import warnings


"""used to stop warnings from showing up in the console"""
warnings.filterwarnings('ignore')
random.seed(17)

"""CONSTANTS THAT ARE USED THROUGHOUT THE MODEL"""
LOOK_BACK = 10  # the number of previous data points used by the model to predict future time series
SPLIT = 0.8  # the percentage of the data that will be used as the test-set
EPOCHS = 100  # the number of training iterations that the model will go through
BATCH_SIZE = 100  # the number of training samples that the model uses in on iteration (epoch)
YEAR = 2000  # start year of the data
TICKER = "^GSPC"  # ticker of the stock (or financial security) that will be analyzed and predicted

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data = yf.download(TICKER, start=f"{YEAR}-01-01")[["Adj Close"]]  # downloads the data
data.columns = ["close"]  # renames the column of the data


def get_CEEMD_residue(data: pd.DataFrame):
    """
    Complete Ensemble EMD with Adaptive Noise (CEEMDAN) performs an EEMD
    The difference is that the information about the noise is shared among all workers

    :returns:
    IMFs : numpy array
        All the Intrinsic Mode Functions that make up the original stock price
    residue : numpy array
        The residue from the recently analyzed stock price
    """

    data_np = data.to_numpy()

    ceemd = CEEMDAN()
    ceemd.extrema_detection = "parabol"
    ceemd.ceemdan(data_np)
    IMFs, residue = ceemd.get_imfs_and_residue()

    nIMFs = IMFs.shape[0]

    plt.figure(figsize=(18, 12))
    plt.subplot(nIMFs + 2, 1, 1)

    plt.plot(data, 'r')
    plt.ylabel("S&P500")

    plt.subplot(nIMFs + 2, 1, nIMFs + 2)
    plt.plot(data.index, residue)
    plt.ylabel("Residue")

    for n in range(nIMFs):
        plt.subplot(nIMFs + 2, 1, n + 2)
        plt.plot(data.index, IMFs[n], 'g')
        plt.ylabel("eIMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=4)

    plt.tight_layout()
    plt.show()

    return IMFs, residue, nIMFs


def plot_IMFs(IMFs: np.ndarray, residue: np.ndarray, num_IMFs: int, data: pd.DataFrame):
    """
    This function aims to reconstruct the Time Series using the IMFs

    :param IMFs: The IMFs returned from using any of the decomposition functions above
    :param residue: The residue returned from using any of the decomposition functions above
    :param num_IMFs: The number of IMFs you want to reconstruct your data. A value of 2 means the last two IMFs
    :return: None
    """

    sum_IMFs = sum(IMFs[-num_IMFs:])
    sum_IMFs += residue

    plt.figure(figsize=(12, 10))
    plt.plot(data.index, data, label="Stock Price")
    plt.plot(data.index, sum_IMFs, label=f"Last {num_IMFs} IMFs")
    plt.legend(loc="upper left")
    plt.show()


def create_dataset(dataset: np.ndarray):
    dataX, dataY = [], []

    for i in range(len(dataset) - LOOK_BACK - 1):
        look_back_data = dataset[i:(i + LOOK_BACK), 0]
        dataX.append(look_back_data)
        dataY.append(dataset[i + LOOK_BACK, 0])

    return np.array(dataX), np.array(dataY)


def LSTM_CNN_CBAM(dataset: np.ndarray, layer: int = 128):

    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))

    # Normalize the data -- using Min and Max values in each subsequence to normalize the values
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    # Split data into training and testing set
    train_size = int(len(dataset) * SPLIT)
    test_size = len(dataset) - train_size
    train, test = dataset[:train_size, :], dataset[train_size:, :]

    trainX, trainY = create_dataset(train)
    testX, testY = create_dataset(test)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM-CNN-CBAM network
    model = Sequential()
    model.add(LSTM(layer, input_shape=(1, LOOK_BACK), return_sequences=True))
    model.add(Conv1D(filters=512, kernel_size=1, activation='relu', input_shape=(1, LOOK_BACK)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_split=0.1)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    testing_error = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    return testPredict, testY, testing_error


def run_model(IMFs):


    IMF_predict_list = []
    error_list = []

    for i, IMF in enumerate(IMFs):
        print(f"IMF number {i + 1}")

        IMF_predict, IMF_test, testing_error = LSTM_CNN_CBAM(IMF, layer=128)
        error_list.append(testing_error)
        IMF_predict_list.append(IMF_predict)

    return IMF_predict_list, error_list



def visualize_results(IMF_predict_list, error_list):
    for i, v in enumerate(IMF_predict_list):
        IMF_predict_list[i] = v[:, 0]

    final_prediction = []
    for i in range(len(IMF_predict_list[0])):

        element = 0

        for j in range(len(IMF_predict_list)):
            element += IMF_predict_list[j][i]

        final_prediction.append(element)

    data_plot = data.close.astype("float32")
    data_plot = np.reshape(data_plot.to_numpy(), (-1, 1))

    train_size = int(len(data_plot) * SPLIT)
    test_size = len(data_plot) - train_size
    data_plot_train, data_plot_test = data_plot[:train_size], data_plot[train_size:]

    data_plot_testX, data_plot_testY = create_dataset(data_plot_test)

    # Calculate the RMSE
    np.sqrt(mean_squared_error(data_plot_testY.tolist(), final_prediction))

    plt.figure(figsize=(18, 12))

    # plot lines
    plt.plot(data.index[train_size + LOOK_BACK + 1:], final_prediction, label="Predicted Value")
    plt.plot(data.index[train_size + LOOK_BACK + 1:], data_plot_testY.tolist(), label="Actual Value")
    plt.legend()
    plt.show()


IMFs, residue, n = get_CEEMD_residue(data["close"])

IMF_predict_list, error_list = run_model(IMFs)

visualize_results(IMF_predict_list, error_list)

