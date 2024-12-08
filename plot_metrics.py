import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, val_accuracies, val_recalls, val_f1s, epochs):
    """
    绘制损失、准确率、召回率和F1指数的变化图
    """
    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制召回率曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_recalls, label='Validation Recall')
    plt.title('Recall Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # 绘制F1指数曲线
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.title('F1 Score Curve')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # 显示所有图形
    plt.tight_layout()
    plt.show()
