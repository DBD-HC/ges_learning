import matplotlib.pyplot as plt
import numpy as np

# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(8, 6), dpi=80)

# 再创建一个规格为 1 x 1 的子图
plt.subplot(1, 1, 1)

# 柱子总数
N = 5
# 包含每个柱子对应值的序列
env_acc = (98.89795918367347, 99.27619047619047, 98.35164835164835, 99.05714285714285, 98.75, 93.90)
env_acc_origin = (98.83673469387755, 98.8, 97.56043956043956, 97.37142857142858, 98.14285714285714, 94.59016393442623)
position_acc = (99.08212560386473, 98.96135265700483, 99.05797101449275, 99.03381642512077, 99.05797101449275)
position_acc_origin = (75.09661835748792, 70.7487922705314, 69.22705314009662, 74.97584541062802, 74.22705314009662)
# 包含每个柱子下标的序列
index = np.arange(N)

# 柱子的宽度
width = 0.35

# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index, position_acc_origin, width, label="origin", color="#87CEFA")
p3 = plt.bar(index+width,  position_acc, width, label="new", color="blue")


# 设置横轴标签
plt.xlabel('positions')
# 设置纵轴标签
plt.ylabel('Accuracy(%)')

# 添加标题
plt.title('position acc ')

envs = ('Meet.', 'Liv.', 'Bed.', 'Off.A.', 'Lab.', 'Off.B.')

loc = ('p1', 'p2', 'p3','p4','p5')

temp = 0
for i in position_acc:
    temp += i

print(temp/5)

# 添加纵横轴的刻度
plt.xticks(index, loc)
plt.ylim(50, 100)

# 添加图例
plt.legend(loc="upper right")
plt.savefig('acc_env.png')
plt.show()