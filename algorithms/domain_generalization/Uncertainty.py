from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt

# 定义思考步骤
steps = ["Thought 1", "Thought 2", "Thought 3", "Answer"]
# 设定不确定性水平（数值越高表示不确定性越大）
uncertainty = [0.6, 0.8, 0.3, 0.1]


# 将离散点转换为平滑曲线
x = np.arange(len(steps))
x_smooth = np.linspace(x.min(), x.max(), 100)
spline = make_interp_spline(x, uncertainty, k=3)  # 三次样条插值
y_smooth = spline(x_smooth)

# 绘制平滑曲线
plt.figure(figsize=(8, 5))
plt.plot(x_smooth, y_smooth, linestyle='-', color='b', label="Uncertainty Level")
plt.scatter(x, uncertainty, color='r', zorder=3)  # 标注原始点

# 设置刻度，使其与原始steps匹配
plt.xticks(x, steps)
plt.title("Smoothed Uncertainty Variation from Thought 1 to Answer")
plt.xlabel("Steps")
plt.ylabel("Uncertainty Level")
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# 显示图表
plt.show()

