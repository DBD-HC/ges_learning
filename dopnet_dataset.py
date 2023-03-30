import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import h5py

root = '/root/autodl-fs'


def read_train_data():
    # Here you must specify the sample you want to see
    person = 'A'
    gesture = '0'
    sample = '1'
    # Load File; Get the path from the file containing the Data
    test = sio.loadmat(os.path.join(root, "Data_Per_PersonData_Training_Person_" + person + ".mat"))

    # Get the gesture;
    x = test["Data_Training"]["Doppler_Signals"][0][0][0][int(gesture)][int(sample)][0]

    # Equation
    x = 20 * np.log10(abs(x) / np.amax(abs(x)))

    # Display Spectogram
    plt.imshow(x, vmin=-50, vmax=0, cmap='jet', aspect='auto', extent=[0, x.shape[1], -501, 500])

    plt.ylabel("Doppler", fontsize=17)
    plt.xlabel("Time", fontsize=17)

    plt.colorbar()
    plt.show()


def read_test_data():
    # Number = input("Please tell me the number of the Sample you want to display: ")

    test = sio.loadmat(sio.loadmat(os.path.join(root, "Data_For_Test_Random.mat")))

    # Get the gesture;
    x = test["Data_rand"][int(1)][0][0][0]

    # Equation
    x = 20 * np.log10(abs(x) / np.amax(abs(x)))

    # Display Spectogram
    plt.imshow(x, vmin=-50, vmax=0, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.show()


read_test_data()