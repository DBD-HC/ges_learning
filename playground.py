import visdom
from sklearn.manifold import TSNE
import torch
import numpy as np
import h5py

vis = visdom.Visdom(env='playground', port=6006)

def tsne_test():
    # 创建一个示例的高维数据集
    data = torch.from_numpy(np.random.rand(100, 50))
    # 创建一个 t-SNE 模型
    tsne = TSNE(n_components=2)  # 降维为2维

    # 对数据运行 t-SNE
    embedded_data = tsne.fit_transform(data)

    # 创建散点图
    scatter = vis.scatter(
        X=embedded_data,  # 数据
        win='manifold',
        opts=dict(
            title='Scatter Plot Example',  # 图表标题
            markersize=5,  # 散点大小
        )
    )



def spiral_test():
    # 生成螺旋曲线的坐标点
    t = np.linspace(0, 10 * np.pi, 1000)  # 生成一千个点
    x = t * np.cos(t)
    y = t * np.sin(t)

    # 创建一个散点图
    vis.scatter(
        X=np.column_stack((x, y)),
        win='diff',
        opts=dict(
            title='Spiral Curve',
            xlabel='X-axis',
            ylabel='Y-axis',
            markersize=5,
            markercolor=np.floor(np.random.rand(1000, 3))
        )
    )

    # 创建一个散点图
    vis.scatter(
        X=np.column_stack((x + 10, y)),
        update='append',
        win='diff',
        opts=dict(
            title='Spiral Curve',
            xlabel='X-axis',
            ylabel='Y-axis',
            markersize=5,
            markercolor=np.floor(np.random.rand(1000, 3))
        )
    )


# spiral_test()


def get_history(filename='envacc.npy'):
    log = np.load(filename)
    print(log)

solid_dataset_root='/root/autodl-tmp/dataset/dsp/'
def get_solid_dataset(file_name = '1_0_0.h5', use_channel = 0):
    # 打开HDF5文件
    # 替换成你的文件路径
    file_path = solid_dataset_root + file_name
    with h5py.File(file_path, 'r') as file:
        # 打印文件中的所有顶层对象
        print("Top-level keys:", list(file.keys()))

        # Data and label are numpy arrays
        data = file['ch{}'.format(use_channel)][()]
        label = file['label'][()]
        print(label)


get_solid_dataset()