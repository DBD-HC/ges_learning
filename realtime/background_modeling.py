import numpy as np


class BackgroundModel:
    def __init__(self, feat_size=(32, 32), num_init_frames=120, num_inter=3):
        self.feat_size = feat_size
        self.background = np.zeros(feat_size)
        self.pre_inter_frame = np.zeros((num_inter, feat_size[-2], feat_size[-1]))
        self.avg_if_diff = np.zeros(feat_size)
        self.std_if_diff = np.zeros(feat_size)
        self.num_init_frames = num_init_frames
        self.num_inter = num_inter
        self.inter_p = 0
        self.thr_factor = 2
        self.lr = 0.1

    def init_background(self, frames):
        if len(frames) < self.num_init_frames:
            print('初始化失败 理想帧数 {} 实际获得的帧数 {}', self.num_init_frames, len(frames))
            return 0
        self.background = np.mean(frames, dtype=0)
        self.pre_inter_frame[:] = frames[-self.num_inter:]
        inter_diff_list = np.zeros((len(frames) - self.num_inter, self.feat_size[-2], self.feat_size[-1]))
        for i in range(self.num_inter, len(frames)):
            inter_diff_list[i] = frames[i] - frames[i - self.num_inter]
        self.std_if_diff = np.std(inter_diff_list, axis=0)
        self.avg_if_diff = np.mean(inter_diff_list, axis=0)


    def get_dynamic_component(self, rai, background):
        # 检测动态目标
        diff = np.abs(rai - background)
        thr = self.avg_if_diff + self.thr_factor * self.std_if_diff
        mask = diff > thr
        dynamic_target = np.zeros(self.feat_size)
        dynamic_target[mask] = 1
        # 更新背景、帧间差平均值和帧间差标准差
        self.background = (1 - self.lr) * self.background + self.lr * diff
        inter_diff = rai - self.pre_inter_frame[self.inter_p]
        self.std_if_diff = (1 - self.lr) * self.std_if_diff + self.lr * inter_diff
        self.avg_if_diff = (1 - self.lr) * self.avg_if_diff + self.lr * np.abs(inter_diff - self.std_if_diff)
        self.pre_inter_frame[self.inter_p] = rai
        self.inter_p = (self.inter_p + 1) % self.num_inter

        return dynamic_target
