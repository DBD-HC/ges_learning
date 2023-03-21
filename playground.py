# Import
import scipy.io
import scipy.signal
import sklearn.cluster
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

gestures = ['Clockwise', 'Counterclockwise', 'Pull', 'Push', 'SlideLeft', 'SlideRight']
negative_samples = ['liftleft', 'liftright', 'sit', 'stand', 'turn', 'walking', 'waving']
envs = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
participants = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13',
                'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25']
locations = ['p1', 'p2', 'p3', 'p4', 'p5']

# Parameters
data = np.load('D:\\dataset\\mmWave_cross_domain_gesture_dataset\\n_liftleft_e1_u1_p1_s1.npy')
print(data)
