import torch


class Debouncer:
    def __init__(self, thr=0.5, finished_thr =0.2, n_gesture_type=7, cache_size =3):
        self.pre_out = torch.zeros(cache_size, n_gesture_type)
        self.p_cache = 0
        self.cache_size = cache_size
        self.thr = thr
        self.finished_thr = finished_thr
        self.gesture_type = 6

    def get_gesture_type(self, x):
        max_index = torch.argmax(x)
        p = x[max_index]

        if p > self.thr:
            all_greater = (self.pre_out[:, p] > self.thr).all()
            if all_greater:
                self.gesture_type = max_index
        elif p < self.finished_thr:
            all_smaller = (self.pre_out[:, :] < self.finished_thr).all()
            if all_smaller:
                res = self.gesture_type
                self.gesture_type = 6
                return res

        return 6


