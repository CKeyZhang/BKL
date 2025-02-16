'''脚本用于获取数据集的可靠性图、期望校准误差（ECE）、
   Brier分数以及测试负对数似然（NLL）值。'''

import numpy as np  # 导入NumPy库用于数组和数学操作
import matplotlib  # 导入Matplotlib用于绘图
import matplotlib.pyplot as plt  # 导入绘图子模块
import torch  # 导入PyTorch库

# 定义命令行参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, required=True,
                    help='模型生成的输出文件名。')
parser.add_argument('--T', type=float, default=1.0, help='用于缩放logits的温度。')
parser.add_argument('--N', type=int, default=10, help='用于ECE测量的箱数。')
parser.add_argument('--num_classes', type=int, default=100,
                    help='任务的输出类别数量。')
parser.add_argument('--visualize', action='store_true', help='是否需要可视化。')

args = parser.parse_args()  # 解析命令行参数

matplotlib.use('Agg')  # 设置Matplotlib的后端为GTK
plt.style.use('ggplot')  # 设置绘图风格为ggplot

file_name = args.file_name  # 从命令行参数获取文件名
N = args.N  # 从命令行参数获取箱数
T = args.T  # 从命令行参数获取温度

# 用于存储每个箱子中正确和观察到的预测计数的数组
bin_array = torch.zeros(N + 1)  # 初始化箱子计数数组
means_array = torch.zeros(N + 1)  # 初始化每个箱子的均值数组
num_array = torch.zeros(N + 1)  # 初始化每个箱子的样本数量数组

brier_score = 0.0  # 初始化Brier分数
total_num = 0.0  # 初始化总样本数量
nll = 0.0  # 初始化负对数似然

# 根据值计算箱子编号
def get_bin_num(val):
    global N
    val = min(int(val * N), N)  # 确保返回的箱子编号在有效范围内
    return val

# 计算softmax函数
def softmax(arr):
    arr1 = arr - torch.max(arr)  # 减去最大值以提高数值稳定性
    return torch.exp(arr1) / torch.sum(torch.exp(arr1))  # 计算softmax并返回结果

probs = torch.empty(0, args.num_classes)

# 读取文件并处理数据
with open(file_name, 'r') as f:  # 打开指定的文件进行读取
    line = f.readline()  # 读取第一行
    while line:  # 当行不为空时继续循环
        if 'Targets' in line:  # 查找包含'Targets'的行
            pos = line.index('Targets:')  # 找到'Targets:'的位置
            line = line[pos + 10:]  # 提取'Targets:'之后的部分
            line = line.replace('][', ' ')  # 替换字符
            line = line.replace(']', '')  # 去除']'
            line = line.replace('[', '')  # 去除'['
            line = line.replace('\n', '')  # 去除换行符
            line = line.replace(',', '')  # 去除逗号
            line = line.split(' ')  # 按空格分割成列表
            line = [int(x) for x in line]  # 转换为整数列表
            xent_index = torch.tensor(line)  # 将列表转换为PyTorch张量
            f.readline()  # 读取下一行
            line = f.readline()  # 读取第一行

        line = line.replace('][', ' ')  # 替换字符
        line = line.replace('],', '')  # 去除'],'
        line = line.replace('[', '')  # 去除'['
        line = line.replace(']]', '')  # 去除']]'
        line = line.replace('\n', '')  # 去除换行符
        line = line.replace(' ', '')  # 去除空格
        line = line.split(',')  # 按逗号分割成列表
        line = [float(x) for x in line]  # 转换为浮点数列表
        line = torch.tensor(line)  # 将列表转换为PyTorch张量
        prob = line.view(-1, args.num_classes)  # 重塑数组形状为(num_samples, num_classes)
        # 使用 torch.cat 添加新的行
        probs = torch.cat((probs, prob), dim=0)
        line = f.readline()  # 读取下一行

    predictions = torch.argmax(probs, dim=1)  # 获取每个样本的预测类别（概率最大者）

print("第一维大小:", predictions.shape[0])

# 计算Brier分数、NLL和箱子中的正确预测
for i in range(xent_index.shape[0]):  # 遍历每个样本
    target = xent_index[i]  # 获取真实标签
    dist = softmax(probs[i] / T)  # 计算当前样本的概率分布
    pred = predictions[i]  # 获取当前样本的预测结果

    ttz = torch.zeros(args.num_classes)  # 初始化目标的one-hot编码
    ttz[target] = 1.0  # 将真实标签位置的值设为1
    brier_score += torch.sum((dist - ttz) ** 2).item()  # 更新Brier分数
    total_num += 1.0  # 更新样本总数

    nll -= torch.log(dist)[target].item()  # 更新负对数似然

    bin_num = get_bin_num(dist[pred])  # 获取预测置信度对应的箱子编号
    means_array[bin_num] += dist[pred]  # 更新均值数组
    num_array[bin_num] += 1.0  # 更新样本数量数组
    if pred == target:  # 如果预测正确
        bin_array[bin_num] += 1.0  # 更新正确预测计数



# 输出计算结果
print('Brier score: ', brier_score / total_num)  # 打印Brier分数
print('NLL: ', nll / total_num)  # 打印负对数似然

# 合并最后一个箱子
means_array[N - 1] += means_array[N]
bin_array[N - 1] += bin_array[N]
num_array[N - 1] += num_array[N]

means_array = means_array / (num_array + 1e-5)  # 计算均值
bin_array = bin_array / (num_array + 1e-5)  # 计算比例

# 计算期望校准误差（ECE）
ece = torch.abs(bin_array - means_array)  # 计算每个箱子的绝对误差
ece = ece * num_array  # 加权绝对误差
ece = torch.sum(ece[:-1]) / torch.sum(num_array[:-1])  # 计算ECE
print('Expected Calibration Error: ', ece.item())  # 打印期望校准误差

# 可视化结果
if args.visualize:  # 如果需要可视化
    plt.figure()  # 创建一个新图形
    plt.gcf().set_facecolor('white')
    plt.plot(means_array[:-1].numpy(), bin_array[:-1].numpy(), linewidth=3.0)  # 绘制均值与箱子比例的关系
    plt.plot(0.1 * np.arange(11), 0.1 * np.arange(11), linewidth=3.0)  # 绘制对角线
    plt.plot(means_array[:-1].numpy(), num_array[:-1].numpy() / torch.sum(num_array[:-1]).item())  # 绘制样本数量的比例
    plt.scatter(means_array[:-1].numpy(), num_array[:-1].numpy() / torch.sum(num_array[:-1]).item())  # 绘制散点图
    plt.text(0.1, 0.9, f'ECE: {ece:.4f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.1, 0.8, f'Brier Score: {brier_score / total_num:.4f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.1, 0.7, f'NLL: {nll / total_num:.4f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.savefig('./output/ece/output.png', dpi=300, bbox_inches='tight', transparent=True)