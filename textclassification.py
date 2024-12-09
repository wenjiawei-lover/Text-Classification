import numpy as np
import pandas as pd
from wordcloud import WordCloud
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from plot_metrics import plot_metrics
import matplotlib.pyplot as plt
from plot_metrics import plot_metrics
# 读取数据集的函数
def load_data(file_path):
    data = pd.read_csv(file_path, header=None, sep='\t')
    texts = data[1].values
    labels = data[0].values
    return texts, labels
# 生成词云的函数
def generate_wordcloud(texts):
    # 合并所有文本
    all_text = ' '.join(texts)

    # 创建词云对象
    wordcloud = WordCloud(font_path='E:/texlive/2023/texmf-dist/fonts/truetype/public/arphic-ttf/gbsn00lp.ttf', width=800, height=400, background_color='white').generate(
        all_text)#请替换自己路径

    # 绘制词云
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 关闭坐标轴
    plt.show()
# 定义数据集
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
# 使用中文BERT模型
def create_model(num_classes):
    bert_model = BertModel.from_pretrained('E:\\project\\Text-Classification\\data')#请替换自己路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)

    class TextClassifier(nn.Module):
        def __init__(self):
            super(TextClassifier, self).__init__()
            self.bert = bert_model
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            x = outputs.last_hidden_state[:, 0, :]  # 取出CLS token
            x = self.dropout(x)
            return self.fc(x)

    return TextClassifier, device


# 训练函数（带混合精度训练）
def train(model, train_loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", ncols=100, leave=False, dynamic_ncols=True)
    for batch in progress_bar:
        optimizer.zero_grad()

        with autocast():  # 使用混合精度训练
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

        # 使用Scaler进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))  # 动态更新 loss
    return total_loss / len(train_loader)#ai修改部分


# 验证函数
def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(val_loader, desc="Validating", ncols=100, leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            pred = outputs.argmax(dim=1).cpu()
            correct += (pred == labels.cpu()).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=val_loss / (progress_bar.n + 1))  # 动态更新 val_loss
    accuracy = correct / len(val_loader.dataset)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss / len(val_loader), accuracy, recall, f1


# 主函数
def main():
    # 设置数据集路径
    data_path = 'E:\\project\\Text-Classification\\data\\'#请替换自己路径
    train_texts, train_labels = load_data(data_path + 'cnews.train.txt')
    val_texts, val_labels = load_data(data_path + 'cnews.val.txt')
    test_texts, test_labels = load_data(data_path + 'cnews.test.txt')

    # 预处理标签
    label_set = {label: idx for idx, label in enumerate(set(train_labels))}
    num_classes = len(label_set)
    train_labels = np.array([label_set[label] for label in train_labels])
    val_labels = np.array([label_set[label] for label in val_labels])
    test_labels = np.array([label_set[label] for label in test_labels])

    # 设置BERT分词器，使用中文的BERT
    tokenizer = BertTokenizer(vocab_file='E:\\project\\Text-Classification\\data\\vocab.txt')#请替换自己路径

    # 文本编码
    def encode_texts(texts, max_length=128):
        return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=max_length,
                         return_tensors='pt')

    train_encodings = encode_texts(train_texts)
    val_encodings = encode_texts(val_texts)
    test_encodings = encode_texts(test_texts)

    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    # 创建模型和设备
    TextClassifier, device = create_model(num_classes)
    model = TextClassifier().to(device)

    # 冻结BERT部分层
    for param in model.bert.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    # 初始化混合精度训练的Scaler
    scaler = GradScaler()

    # 训练和验证
    train_losses = []
    val_accuracies = []
    val_losses = []
    val_recalls = []
    val_f1s = []
    val_recall_loss = []  # 记录损失率

    for epoch in range(20):  # 训练20次
        print(f"\nEpoch {epoch + 1}")
        train_loss = train(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_accuracy, val_recall, val_f1 = validate(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_recall_loss.append(val_loss)  # 记录每次的损失

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

    # 打印测试集的性能
    test_loss, test_accuracy, test_recall, test_f1 = validate(model, test_loader, loss_fn, device)
    print(
        f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')

    # 打印分类报告
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=list(label_set.keys())))

    # 生成词云
    generate_wordcloud(train_texts)
 # 在训练和验证完成后，调用 plot_metrics 绘制图表
    plot_metrics(train_losses, val_losses, val_accuracies, val_recalls, val_f1s, list(range(1, 21)))

# 运行训练和评估
if __name__ == "__main__":
    main()