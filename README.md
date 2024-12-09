# Text-Classification
# 文本分类项目

## 概述
本项目是一个使用BERT模型构建的文本分类系统，能够将文本分类到预定义的类别中。系统包括数据预处理、模型训练、评估和指标可视化。

## 环境要求
要运行此项目，您需要安装以下包。您可以使用`pip`安装它们：

```
pip install -r requirements.txt
```

所需的包包括：
- matplotlib==3.9.3
- numpy==2.1.3
- pandas==2.2.3
- scikit-learn==1.5.2
- torch==2.5.0+cu118
- tqdm==4.67.1
- transformers==4.47.0
- wordcloud==1.9.4

## 安装步骤
1. 确保您的系统已安装Python。
2. 安装`requirements.txt`中列出的所需包。
3. 将项目文件放置在您选择的目录中。

## 数据
项目期望数据以特定格式存在：
- `cnews.train.txt`：训练数据集。
- `cnews.val.txt`：验证数据集。
- `cnews.test.txt`：测试数据集。

这些文件应放置在项目根目录下的`data`目录中。

## 使用方法
要运行项目，请执行`textclassification.py`脚本：

```bash
python textclassification.py
```

### 主要功能
- **数据加载和预处理**：`load_data`函数读取数据集并将文本和标签分开。`encode_texts`函数使用BERT的分词器对文本进行分词。
- **生成词云**：`generate_wordcloud`函数从训练文本创建词云，提供最频繁词汇的视觉表示。
- **自定义数据集**：`TextDataset`类是一个自定义数据集类，它包装了编码后的文本和标签，以便与PyTorch的DataLoader一起使用。
- **BERT模型**：`create_model`函数初始化一个用于文本分类的BERT模型。
- **训练和验证**：`train`和`validate`函数分别处理训练和验证过程，使用PyTorch的DataLoader和自定义进度条进行跟踪。
- **混合精度训练**：项目使用PyTorch的`GradScaler`进行混合精度训练，以加快训练速度并减少内存使用。
- **指标绘图**：`plot_metrics`函数来自`plot_metrics.py`，可视化训练和验证损失、准确率、召回率和F1分数。
- **分类报告**：训练后，生成分类报告，提供模型性能的详细指标。

## 可视化
项目包括关键指标的可视化：
- 损失曲线
- 准确率曲线
- 召回率曲线
- F1分数曲线

这些使用matplotlib绘制，并在训练期间每个epoch后显示。

## 输出
- 项目输出每个epoch的训练损失、验证损失、准确率、召回率和F1分数。
- 训练结束后，打印测试损失、准确率、召回率和F1分数。
- 为测试数据集生成分类报告，提供模型性能的详细分析。
##数据
![微信截图_20241209144315](https://github.com/user-attachments/assets/d186b58b-c332-4de4-ab63-94d5a99d54e4)
![微信截图_20241209144330](https://github.com/user-attachments/assets/6466b451-fa11-46ce-9591-b6e0dd9b0ae3)

## 贡献
欢迎通过点star，提交fork请求或在GitHub上开启问题来为这个项目贡献。

## 许可证
本项目是开源的，并在[MIT许可证](LICENSE)下可用。
