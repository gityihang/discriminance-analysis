import torch

import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from models.Model import ClaModel

data_train = pd.read_csv('byh/src/datasets/winequality-red_train.csv')
data_test = pd.read_csv('byh/src/datasets/winequality-red_test.csv')

data_train = data_train.sample(frac=1).reset_index(drop=True)  #打乱数据所在的行

device = torch.device("cuda")

inputs_train= data_train.drop(columns='quality_two').values
outputs_train = data_train[["quality_two"]].values

model = ClaModel(11,32,1).to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(),0.001)

num_epochs = 800

# 将测试集数据加载到设备上  
inputs_test = torch.Tensor(data_test.drop(columns='quality_two').values).to(device)  
outputs_test = torch.Tensor(data_test[["quality_two"]].values).to(device)  
  
# 用于存储训练集和测试集损失的列表  
train_losses = []  
test_losses = []  

for epoch in range(num_epochs):
    # 将输入和标签加载到设备上
    inputs = torch.Tensor(inputs_train).to(device)
    targets = torch.Tensor(outputs_train).to(device)
    
    # 通过模型生成预测
    predictions = model(inputs)
    
    # 计算损失
    loss = criterion(predictions.squeeze(), targets.float().squeeze())
    
    # 清零梯度，反向传播和更新模型参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    
    # 每100轮输出一次训练结果
    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}")

        # 计算测试集损失  
        model.eval()  # 设置模型为评估模式  
        with torch.no_grad():  # 不需要计算梯度  
            test_predictions = model(inputs_test)  
            test_loss = criterion(test_predictions.squeeze(), outputs_test.float().squeeze())  
        test_losses.append(test_loss.item())  
        model.train()  # 设置模型回训练模式  
        print(f"Test Loss: {test_loss.item():.4f}")  

 # 绘制损失曲线  
plt.figure(figsize=(10, 5))  
plt.plot(train_losses, label='Training Loss')  
plt.plot(test_losses, label='Test Loss')  
plt.title('Loss over Epochs')  
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
plt.legend()  
plt.show()   

# 保存模型
torch.save(model.state_dict(), "/home/byh/yanjiusheng/xiangmu/判别分析作业/discriminance-analysis/byh/checkpoint/Cla_model-windows.pth")




