import os.path

import numpy as np
import torch
from data.mcd_dataset import *
from model.network import RAIRadarGestureClassifier
from visual.feat_visualizer import pca_visualize, tsne_visualize, feat_heatmap


def get_model(need_cfar=True, ra_conv=True, diff=True, path='../cross_pos0_aug_False_diff_True_ra_True.pth'):
    net = RAIRadarGestureClassifier(cfar=need_cfar, track=False, spatial_channels=(4, 8, 16), ra_conv=ra_conv,
                                    heads=4,
                                    track_channels=(4, 8, 16), track_out_size=32, hidden_size=(128, 128), diff=diff,
                                    ra_feat_size=32, attention=True, cfar_expand_channels=8, in_channel=1)
    net.load_state_dict(torch.load(path)['model_state_dict'])
    net.eval()
    net = net.to(device)
    return  net

def run_model(rdi, model=None):
    data_len = len(rdi)
    rdi = torch.from_numpy(rdi[None, :]).type(torch.float32).to(device)
    data_len = torch.Tensor([data_len]).type(torch.float32).to(device)
    if model is None:
        model = get_model()
    with torch.no_grad():
        res = model(rdi, data_len)
    # ges_type = torch.argmax(res, 1).item()
    return res

def get_data(act='y_Pull', env='e1', user='u4', pos='p1', sample=1):
    file_format = '{}_{}_{}_{}_s{}.npy'
    filename = file_format.format(act, env, user, pos, sample)
    filename = rai_data_root + '/' + filename
    if os.path.exists(filename):
        file = np.load(filename)
        file = data_normalization(file)
        return file
    return None

def ges_classify():
    res = run_model(get_data())
    ges_type = torch.argmax(res, 1).item()
    print(ges_type)

def visual_diff_hidden_feats(act='y_SlideLeft'):
    model = get_model()
    sn = model.frame_model
    sn.need_diff = False
    range_m = sn.range_conv
    range_m.need_diff = False
    model = sn
    position1_list = []
    position2_list = []
    feat1_list = None
    feat2_list = None
    for user in participants:
        for pos in locations:
            s = 1
            rdi = get_data(act=act, user=user, pos=pos, sample=s)
            while rdi is not None:
                res = run_model(rdi, model)
                print(res)
                pca_visualize(res[0], win=act+pos, clazz=pos, all_clazz=locations, title=act+pos)
                res = run_model(get_data(act=act, pos=pos), lambda rdi, data_len: range_m(torch.squeeze(rdi)[:, None, :],
                                                                                              rdi.size(0), rdi.size(1)))
                sn = model.frame_model
                sn_diff = sn.diff
                feat1 = torch.squeeze(sn_diff.featmap1).detach().cpu().numpy()
                if feat1_list is not None:
                    feat1_list = np.concatenate((feat1_list, feat1), axis=0)
                else:
                    feat1_list = feat1
                feat2 = torch.squeeze(sn_diff.featmap2).detach().cpu().numpy()
                if feat2_list is not None:
                    feat2_list = np.concatenate((feat2_list, feat2), axis=0)
                else:
                    feat2_list = feat2
                position1_list.extend([int(pos[1:])] * len(feat1))
                position2_list.extend([int(pos[1:])] * len(feat2))
                s += 1
                rdi = get_data(act=act, pos=pos, sample=s)

    pca_visualize(feat1_list, win=act + ' before diff1', clazz=position1_list, all_clazz=locations,
                  title=act + ' before diff')
    pca_visualize(feat2_list, win=act + ' after diff1', clazz=position2_list, all_clazz=locations,
                  title=act + ' after diff')


    #
    #     diff = range_m.diff
    #     res = run_model(get_data(act=act, pos=pos), lambda rdi, data_len: diff(res, rdi.size(0), rdi.size(1)))
    #     pca_visualize(res, win=act + pos + 'diff', clazz=pos, all_clazz=locations, title=act+pos+'diff')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    visual_diff_hidden_feats()
