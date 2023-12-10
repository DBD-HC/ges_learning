import torch
import visdom
from torchvision import models
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

vis = visdom.Visdom(env='feat visual', port=6006)


def plot2(win, pca_result, clazz, title, all_clazz):
    vis.line(X=np.arange(len(pca_result)), Y=pca_result[:, 0], win=win + 'x',
             opts=dict(title=title + 'comp1', xlabel='frames', ylabel='amp'))
    vis.line(X=np.arange(len(pca_result)), Y=pca_result[:, 1], win=win + 'y',
             opts=dict(title=title + 'comp2', xlabel='frames', ylabel='amp'))


def plot(win, pca_result, clazz, title, all_clazz):
    if not isinstance(clazz, list):
        Y = np.ones(len(pca_result)) * int(clazz[1:])
    else:
        Y = clazz
    if not vis.win_exists(win):
        vis.scatter(
            win=win,
            X=pca_result,  # 以二维数组形式传入数据
            Y=Y,
            opts=dict(
                width=600,
                height=400,
                title=title,
                legend=all_clazz,  # 图例
                markersize=5,  # 散点大小
                # markercolor=legend_colors
                # markercolor=np.random.randint(0, 255, size=(10, 3)).tolist()  # 散点颜色
            ),
        )
    else:
        vis.scatter(
            X=pca_result,
            Y=Y,
            win=win,
            name=clazz,
            update='update',
            opts=dict(
                width=800,
                height=600,
                title=win,
                markersize=5,  # 散点大小
            )
        )


def tsne_visualize(feat, win, clazz, all_clazz, title):
    feat = torch.squeeze(feat)
    feat = feat.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    tsne_results = tsne.fit_transform(feat.reshape(feat.shape[0], -1))
    plot(win, tsne_results, clazz, title, all_clazz)


def pca_visualize(feat, win, clazz, all_clazz, title):
    # feat = torch.squeeze(feat)
    # feat = feat.detach().cpu().numpy()
    # 进行 PCA 降维
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(feat.reshape(feat.shape[0], -1))
    plot(win, pca_result, clazz, title, all_clazz)


def feat_heatmap(feat, win, clazz, all_clazz, title):
    feat = torch.squeeze(feat)
    feat = feat.detach().cpu().numpy()
    feat = feat.reshape(feat.shape[0], -1)
    vis.heatmap(
        X=feat,
        opts=dict(
            xlabel='frames',
            ylabel='feats',
            title=title
        )
    )


def plot_arrow(win, pca_result, clazz, title, all_clazz):
    # 创建 Matplotlib 图表并添加箭头
    plt.figure()
    plt.scatter(pca_result[:, 0], pca_result[:, 1])  # 绘制散点图
    # 添加箭头
    pre_point = pca_result[0]
    for point in pca_result[1:]:
        arrow_start = (pre_point[0], pre_point[1])  # 箭头起点（这里假设箭头从第一个点开始）
        arrow_end = (point[0], point[1])  # 箭头终点（这里假设箭头指向第二个点）
        plt.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
                  width=0.01, color='red')  # 添加箭头
        pre_point = point

    plt.title('Scatter Plot with Arrow')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 保存 Matplotlib 图表为图片
    plt.savefig('scatter_with_arrow.png')

    # 在 Visdom 中显示 Matplotlib 图片
    vis.image(
        np.transpose(plt.imread('scatter_with_arrow.png'), (2, 0, 1)),
        opts=dict(title='Scatter Plot with Arrow'),
        win=win,
    )

    plt.close()
