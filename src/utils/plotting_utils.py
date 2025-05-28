"""
Helper module used for plotting common plots
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.histplot(np.log(rmse))
mean_rmse = np.log(rmse).mean()
max_rmse = np.log(rmse[worst_rmse_idx])
min_rmse = np.log(rmse[best_rmse_idx])
std_rmse = np.log(rmse).std()
sem_rmse = std_rmse / np.sqrt(len(rmse))

plt.vlines(ymin=0, ymax=230, x=mean_rmse, colors="r", linestyles="--", lw = 1.5)
plt.vlines(ymin=0, ymax=230, x=max_rmse, colors="r", linestyles="--", lw = 1.5)
plt.vlines(ymin=0, ymax=230, x=min_rmse, colors="r", linestyles="--", lw = 1.5)

plt.vlines(ymin=0, ymax=150, x=mean_rmse-std_rmse, colors="purple", linestyles="--", lw = 1.5)
plt.vlines(ymin=0, ymax=150, x=mean_rmse+std_rmse, colors="purple", linestyles="--", lw = 1.5)
plt.hlines(xmin=mean_rmse-std_rmse, xmax=mean_rmse, y=100, colors="purple", linestyles="--", lw = 1.5)
plt.hlines(xmin=mean_rmse+std_rmse, xmax=mean_rmse, y=100, colors="purple", linestyles="--", lw = 1.5)

plt.annotate(text=f"mean rmse is {mean_rmse.round(0)}",
            xy=(mean_rmse, 210),
            xytext=(mean_rmse + 0.5, 210),
            arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.annotate(text=f"max rmse is {max_rmse.round(0)}",
            xy=(max_rmse, 210),
            xytext=(max_rmse + 0.5, 210),
            arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.annotate(text=f"min rmse is {min_rmse.round(0)}",
            xy=(min_rmse, 210),
            xytext=(min_rmse + 0.5, 210),
            arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.xlabel("$log(RMSE)$")

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.title("Error Distribution", loc="left", fontsize=15)

plt.show()

