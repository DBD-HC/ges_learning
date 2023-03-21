# Import
import scipy.io
import scipy.signal
import sklearn.cluster
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2


# Parameters
n_antenna = 4
n_frame_per_antenna = 400
n_chirp_per_frame = 128
n_sample_per_chirp = 512
d_antenna = 0.5

n_angle = 180
alpha_spacial_cov = 0.01

n_guard_cell_range = 8   # number of guard cells along range dimension
n_train_cell_range = 24  # number of train cells along range dimension
exp_pfa_range = 1e-3     # expected probability of false alarm

n_guard_cell_angle = 8   # number of guard cells along angle dimension
n_train_cell_angle = 24   # number of train cells along angle dimension
exp_pfa_angle = 1e-3     # expected probability of false alarm

dbscan_eps = 3
dbscan_min_samples = 2

# Read data
raw_data = scipy.io.loadmat("data/PeopleWalking/1.mat")["a"]
raw_data = raw_data.T.reshape((n_antenna, n_frame_per_antenna, n_chirp_per_frame, n_sample_per_chirp))


# Range FFT
range_fft_raw = np.fft.fft(raw_data, axis=3)
# Static Clutter Removal
range_fft = range_fft_raw - range_fft_raw.mean(axis=2).reshape((n_antenna, n_frame_per_antenna, 1, n_sample_per_chirp))
# Doppler FFT
doppler_fft = np.fft.fft(range_fft, axis=2)
# Angle vector
angle_vector = np.arange(n_angle) / n_angle * np.pi - np.pi / 2
angle_vector = np.sin(angle_vector).reshape((-1, 1)) @ np.arange(n_antenna).reshape((1, -1))
angle_vector = np.exp(2j * np.pi * d_antenna * angle_vector)
# Generate range-azimuth heatmap using CAPON
range_azimuth = np.empty((n_frame_per_antenna, n_sample_per_chirp, n_angle), dtype=complex)
for frame_idx in tqdm(range(n_frame_per_antenna)):
    for range_idx in range(n_sample_per_chirp):
        x = range_fft[:, frame_idx, :, range_idx]
        r = x @ x.conj().T / 128
        r += alpha_spacial_cov * np.trace(r) / n_antenna * np.eye(n_antenna)
        r = np.linalg.inv(r)
        for angle_idx in range(n_angle):
            a = angle_vector[angle_idx].reshape((-1, 1))
            range_azimuth[frame_idx, range_idx, angle_idx] = 1 / (a.conj().T @ r @ a)[0, 0]


# CFAR along range dimension
kernel_range = np.ones((n_guard_cell_range + n_train_cell_range) * 2 + 1)
kernel_range[n_train_cell_range:n_train_cell_range+2*n_guard_cell_range+1] = 0
kernel_range_sum = kernel_range.sum()
alpha_range = n_train_cell_range * 2 * (exp_pfa_range ** (-1 / (n_train_cell_range * 2)) - 1)
cfar_range = np.empty_like(range_azimuth, dtype=bool)
for frame_idx in tqdm(range(n_frame_per_antenna)):
    for angle_idx in range(n_angle):
        data = np.abs(range_azimuth[frame_idx, :, angle_idx])
        threshold = alpha_range * scipy.signal.convolve(data, kernel_range, mode="same") / kernel_range_sum
        cfar_range[frame_idx, :, angle_idx] = data > threshold

# CFAR along angle dimension
kernel_angle = np.ones((n_guard_cell_angle + n_train_cell_angle) * 2 + 1)
kernel_angle[n_train_cell_angle:n_train_cell_angle+2*n_guard_cell_angle+1] = 0
kernel_angle_sum = kernel_angle.sum()
alpha_angle = n_train_cell_angle * 2 * (exp_pfa_angle ** (-1 / (n_train_cell_angle * 2)) - 1)
cfar_angle = np.empty_like(range_azimuth, dtype=bool)
for frame_idx in tqdm(range(n_frame_per_antenna)):
    for range_idx in range(n_sample_per_chirp):
        data = np.abs(range_azimuth[frame_idx, range_idx, :])
        threshold = alpha_angle * scipy.signal.convolve(data, kernel_angle, mode="same") / kernel_angle_sum
        cfar_angle[frame_idx, range_idx, :] = data > threshold

# 2-pass CFAR
range_azimuth_cfar = np.logical_and(cfar_range, cfar_angle)

range_azimuth_dbscan = range_azimuth_cfar.copy()
for frame_idx in tqdm(range(n_frame_per_antenna)):
    data = range_azimuth_dbscan[frame_idx]
    # Coordinate transform
    row, col = np.where(data)
    radius = n_sample_per_chirp - row
    theta = np.deg2rad(col / n_angle * 180)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points = np.stack([x, y]).T
    # DBSCAN
    if len(points) == 0:
        continue
    dbscan = sklearn.cluster.DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    result = dbscan.fit(points).labels_
    # Remove all points outside the largest cluster
    largest_cluster = np.bincount(result + 1).argmax() - 1
    remove = result != largest_cluster
    data[row[remove], col[remove]] = False

# Doppler FFT on each object in range-azimuth heatmap
velocity_range_azimuth = np.zeros((n_frame_per_antenna, n_sample_per_chirp, n_angle), dtype=np.int64)
for frame_idx in tqdm(range(n_frame_per_antenna)):
    for range_idx in range(n_sample_per_chirp):
        if not np.any(range_azimuth_dbscan[frame_idx, range_idx]):
            continue
        x = range_fft[:, frame_idx, :, range_idx]
        r = x @ x.conj().T / 128
        r += alpha_spacial_cov * np.trace(r) / n_antenna * np.eye(n_antenna)
        r = np.linalg.inv(r)
        for angle_idx in range(n_angle):
            if not range_azimuth_dbscan[frame_idx, range_idx, angle_idx]:
                continue
            a = angle_vector[angle_idx].reshape((-1, 1))
            p = 1 / (a.conj().T @ r @ a)[0, 0]
            w = r @ a / p
            wx = (w.T @ x)[0]
            v = np.abs(np.fft.fft(wx))
            v = np.roll(v, shift=int(len(v)/2))
            velocity_range_azimuth[frame_idx, range_idx, angle_idx] = v.argmax() - int(len(v)/2)


def img2video(path, count=n_frame_per_antenna, height=720, width=720, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for idx in range(count):
        img = cv2.imread(f"fig/tmp/{idx}.jpg")
        video.write(img)
    video.release()

def show_range_fft(range_fft, path, title):
    data = np.mean(np.abs(range_fft), axis=0)
    data = data.reshape((n_frame_per_antenna*n_chirp_per_frame, n_sample_per_chirp))
    data = data.T[int(n_sample_per_chirp/2):]
    data = cv2.resize(data, (512, 512), interpolation=cv2.INTER_AREA)
    plt.figure(figsize=(10, 10))
    # plt.title(title)
    plt.imshow(data, vmax=1e4, cmap="Blues", extent=[0,16,0,12.4])
    plt.xlabel("Time/s", fontsize=16)
    plt.ylabel("Distance/m", fontsize=16)
    plt.savefig(path)
    plt.close()

def show_doppler_fft(doppler_fft):
    for frame_idx in tqdm(range(n_frame_per_antenna)):
        data = np.mean(np.abs(doppler_fft[:, frame_idx]), axis=0)
        data = data.T[int(n_sample_per_chirp/2):]
        data = data[:, 1:]
        data = np.roll(data, shift=int(n_chirp_per_frame/2), axis=1)
        data = data[:, ::-1]
        plt.figure(figsize=(10, 10))
        # plt.title(f"Doppler FFT")
        plt.imshow(data, vmax=1e5, cmap="Blues", extent=[-4,4,0,12.4])
        plt.xlabel("Speed/m/s", fontsize=16)
        plt.ylabel("Distance/m", fontsize=16)
        plt.savefig(f"fig/tmp/{frame_idx}.jpg")
        plt.close()
    img2video(f"fig/doppler_fft.mp4")

def plot_heatmap(heatmap, name, vmin, vmax, title, cmap="Blues"):
    os.makedirs("fig/tmp", exist_ok=True)
    x = np.deg2rad(np.arange(n_angle))
    y = np.arange(int(n_sample_per_chirp/2))
    for frame_idx in tqdm(range(n_frame_per_antenna)):
        data = np.abs(heatmap[frame_idx, int(n_sample_per_chirp/2):][::-1])
        fig = plt.figure(figsize=(10, 10), dpi=72)
        ax = fig.add_subplot(111, polar=True)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.grid(False)
        ax.pcolormesh(x, y, data, shading="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        # plt.title(title)
        plt.savefig(f"fig/tmp/{frame_idx}.jpg")
        plt.close()
    img2video(f"fig/{name}.mp4")

os.makedirs("fig/tmp", exist_ok=True)

show_range_fft(range_fft_raw, "fig/range_fft_raw.jpg", "Range FFT")
show_range_fft(range_fft, "fig/range_fft.jpg", "Range FFT (after static clutter removal)")

show_doppler_fft(doppler_fft)

plot_heatmap(range_azimuth, "range_azimuth", vmin=0, vmax=1e6, title="Range Azimuth")
plot_heatmap(range_azimuth_cfar, "range_azimuth_cfar", vmin=0, vmax=1, title="Range Azimuth (after CFAR)")
plot_heatmap(range_azimuth_dbscan, "range_azimuth_dbscan", vmin=0, vmax=1, title="Range Azimuth (after CFAR & DBSCAN)")
plot_heatmap(velocity_range_azimuth+20, "velocity_range_azimuth", vmin=0, vmax=40, title="Velocity Range Azimuth", cmap="RdBu")