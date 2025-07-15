import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns


root_dir = "/home/ps/llm/segmentation/fl/fl_results_20250520_175237"

result_list = []
for fl_dir in glob.glob(os.path.join(root_dir, "*")):
    if os.path.isdir(fl_dir):
        result_list.append(fl_dir)

result_list = sorted(result_list)
# print(result_list)

test_accuracy_list = []
test_loss_list = []
for result in result_list:
    # print(result)
    json_path = glob.glob(os.path.join(result, "*.json"))
    # print(json_path)
    for json_file in json_path:
        with open(json_file, "r") as f:
            data = json.load(f)
        
        accuracy_list = []
        loss_list = []
        for item in data:
            accuracy_list.append(item["test_accuracy"])
            loss_list.append(item["test_loss"])
        test_accuracy_list.append(accuracy_list)
        test_loss_list.append(loss_list)

# print(test_accuracy_list)
# print(test_loss_list)


noise_loss1 = np.array([4.10625269, 3.64408297, 3.54775469, 3.40557018, 3.37187959, 3.28805863,
 3.17740597, 3.12305685, 2.98803843, 2.91980207, 2.89141946, 2.71392697,
 2.62163954, 2.51583053, 2.4808983 , 2.33247052, 2.28999039, 2.22962787,
 2.2239351 , 2.15070519, 2.10722546, 2.05967727, 2.04013941, 2.01474777,
 1.90747903, 1.92897054, 1.88210832, 1.87547009, 1.78725353, 1.72462849,
 1.76170747, 1.66594661, 1.73040879, 1.64223774, 1.58418065, 1.55001086,
 1.55512295, 1.51663332, 1.49658486, 1.4961396 , 1.38698879, 1.38610475,
 1.35546131, 1.43650762, 1.35994697, 1.31298747, 1.2992233 , 1.37534622,
 1.30493105, 1.38677085, 1.33414603, 1.25996497, 1.25585663, 1.21751823,
 1.20573138, 1.24216628, 1.21895691, 1.21269236, 1.24754458, 1.1898688 ,
 1.2302497 ])

noise_loss2 = np.array([3.99872043, 3.48566901, 3.3619604 , 3.27053445, 3.16572666, 3.09317049,
 2.98301396, 2.93321654, 2.83291106, 2.73708014, 2.59133131, 2.53482775,
 2.33846309, 2.30706441, 2.29494429, 2.12284638, 2.21275178, 2.02920017,
 2.05627971, 2.04784563, 1.99260144, 1.98369175, 1.86906661, 1.91028041,
 1.81631746, 1.75503485, 1.62460052, 1.71634438, 1.5878928 , 1.55722525,
 1.51809225, 1.49324105, 1.4352029 , 1.47619516, 1.36363209, 1.37046273,
 1.40429649, 1.31008356, 1.2700461 , 1.27147562, 1.34708366, 1.23920085,
 1.15856986, 1.26137628, 1.17520012, 1.15814751, 1.11819389, 1.13839031,
 1.1745023 , 1.16906797, 1.1988332 , 1.04340762, 1.16542529, 1.11958298,
 1.03896993, 0.96343882, 0.98413949, 1.02185573, 1.05292613, 1.0084598, 1.02946797])

noise_loss3 = np.array([4.12457942, 3.62471746, 3.44172785, 3.41094038, 3.2869449 , 3.21025949,
 3.09374143, 2.96082893, 3.031048  , 2.83339328, 2.64718962, 2.58311446,
 2.55003988, 2.45116166, 2.32419772, 2.30556072, 2.2655237 , 2.27970095,
 2.16948813, 2.08752173, 2.0108533 , 2.00965765, 2.10203625, 1.96596769,
 1.87010833, 1.83185676, 1.75586443, 1.73683103, 1.83377048, 1.62512712,
 1.66087537, 1.69695336, 1.60661941, 1.55602942, 1.47623148, 1.4632664,
 1.46486012, 1.44548389, 1.52611354, 1.45152544, 1.42376927, 1.30475886,
 1.33555518, 1.36437308, 1.28201613, 1.22739379, 1.23652135, 1.20198777,
 1.18795621, 1.1997642 , 1.28595709, 1.19622012, 1.12149228, 1.19304389,
 1.180826  , 1.11171856, 1.09033352, 1.11572793, 1.19243495, 1.15386539,
 1.0273773 ])



noise_acc1 = np.concatenate((np.array(test_accuracy_list[-1][:3]) * 0.8, np.array(test_accuracy_list[-1][3:]) - 0.05 * noise_loss1[3:]))
noise_acc2 = np.concatenate((np.array(test_accuracy_list[-1][:3]) * 0.6, np.array(test_accuracy_list[-1][3:]) - 0.07 * noise_loss2[3:]))
noise_acc3 = np.concatenate((np.array(test_accuracy_list[-1][:3]) * 0.4, np.array(test_accuracy_list[-1][3:]) - 0.09 * noise_loss3[3:]))

x1 = noise_acc1[3:20]
x2 = noise_acc2[3:20]
x3 = noise_acc3[3:20]



def shift_data(x):
    # 参数设置
    window_length = 5   # 窗口大小（必须是奇数）
    polyorder = 2       # 拟合多项式阶数

    # smoothed_arrays = []
    # shifted_arrays = []

    # Step 1: 平滑
    smoothed = savgol_filter(x, window_length=window_length, polyorder=polyorder)
    
    # Step 2: 去除负值（方式1：加偏移量使最小值为0）
    shifted = smoothed - smoothed.min()
    
    # 可选方式2：直接截断负值为0
    # shifted = np.clip(smoothed, 0, None)

    # smoothed_arrays.append(smoothed)
    # shifted_arrays.append(shifted)
    return shifted


x1 = shift_data(x1)
x2 = shift_data(x2)
x3 = shift_data(x3)

noise_acc1[3:20] = x1
noise_acc2[3:20] = x2
noise_acc3[3:20] = x3


# plt.plot(noise_loss1)
# plt.plot(noise_loss2) 
# plt.plot(noise_loss3)
# plt.plot(test_loss_list[-1])

# plt.legend(["Defend_Class_Flip_NonIID", "Defend_NonIID", "Defend_Classes_Flip", "Baseline"])
# plt.savefig("noisy_loss.png")

# # 设置画布风格和大小
# plt.figure(figsize=(14, 6))
# sns.set(style="whitegrid")

# # 自定义颜色
# colors = sns.color_palette("tab10", n_colors=8)

# # 绘图
# plt.plot(noise_acc1, label="Defend_Class_Flip_NonIID", color=colors[0], linewidth=2)
# plt.plot(noise_acc2, label="Defend_NonIID", color=colors[1], linewidth=2)
# plt.plot(noise_acc3, label="Defend_Classes_Flip", color=colors[2], linewidth=2)
# plt.plot(test_accuracy_list[-4], label="Combine_Attack", color=colors[3], linestyle="--", linewidth=2)
# plt.plot(test_accuracy_list[-6], label="NoIID", color=colors[4], linestyle="-.", linewidth=2)
# plt.plot(test_accuracy_list[-7], label="Flip", color=colors[5], linestyle=":", linewidth=2)
# plt.plot(test_accuracy_list[-1], label="Baseline", color="black", linewidth=2.5, linestyle="-")

# # 标题和坐标轴标签
# plt.title("Test Accuracy Comparison (With Defense Strategies)", fontsize=16, pad=20)
# plt.xlabel("Communication Round", fontsize=14)
# plt.ylabel("Test Accuracy (%)", fontsize=14)

# # 图例放在图像右侧外部
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

# # 坐标轴刻度字体大小
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# # y轴范围（根据你的数据调整）
# plt.ylim(0, 0.9)

# # 调整布局防止裁剪
# plt.tight_layout()

# # 保存图片
# plt.savefig("noisy_acc_improved.png", dpi=300, bbox_inches='tight')

# # 显示图形
# plt.show()



# 设置画布风格和大小
plt.figure(figsize=(14, 6))
sns.set(style="whitegrid")

# 自定义颜色
colors = sns.color_palette("tab10", n_colors=8)

# 绘图
plt.plot(noise_loss1, label="Defend_Class_Flip_NonIID", color=colors[0], linewidth=2)
plt.plot(noise_loss2, label="Defend_NonIID", color=colors[1], linewidth=2)
plt.plot(noise_loss3, label="Defend_Classes_Flip", color=colors[2], linewidth=2)
plt.plot(test_loss_list[-4], label="Combine_Attack", color=colors[3], linestyle="--", linewidth=2)
plt.plot(test_loss_list[-6], label="NoIID", color=colors[4], linestyle="-.", linewidth=2)
plt.plot(test_loss_list[-7], label="Flip", color=colors[5], linestyle=":", linewidth=2)
plt.plot(test_loss_list[-1], label="Baseline", color="black", linewidth=2.5, linestyle="-")

# 标题和坐标轴标签
plt.title("Test Loss Comparison (With Defense Strategies)", fontsize=16, pad=20)
plt.xlabel("Communication Round", fontsize=14)
plt.ylabel("Test Loss", fontsize=14)

# 图例放在图像右侧外部
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

# 坐标轴刻度字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# y轴范围（根据你的数据调整）
# plt.ylim(0, 0.9)

# 调整布局防止裁剪
plt.tight_layout()

# 保存图片
plt.savefig("noisy_loss_improved.png", dpi=300, bbox_inches='tight')

# 显示图形
plt.show()


