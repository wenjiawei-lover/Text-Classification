import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics(train_losses, val_losses, val_accuracies, val_recalls, val_f1s, epochs):
    """
    绘制训练损失、验证损失、验证准确率、验证召回率和验证F1分数的变化图
    """
    # 设置Seaborn的样式为darkgrid
    sns.set_style("darkgrid")

    plt.figure(figsize=(20, 12))

    # 定义颜色和线型
    colors = ['blue', 'orange', 'green', 'red']
    linestyles = ['-', '--', '-.', ':']

    # 绘制训练损失和验证损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color=colors[0], linestyle=linestyles[0], linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color=colors[1], linestyle=linestyles[1], linewidth=2)
    plt.title('Loss Over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)

    # 绘制验证准确率和召回率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color=colors[2], linestyle=linestyles[2], linewidth=2)
    plt.plot(epochs, val_recalls, label='Validation Recall', color=colors[3], linestyle=linestyles[3], linewidth=2)
    plt.title('Accuracy and Recall Over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Metrics', fontsize=12)
    plt.legend(fontsize=10)

    # 绘制验证F1分数曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_f1s, label='Validation F1 Score', color=colors[0], linestyle='-', linewidth=2)
    plt.title('F1 Score Over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(fontsize=10)

    # 设置整个图形的主标题
    plt.suptitle('Training and Validation Metrics', fontsize=16)

    # 调整子图之间的间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()