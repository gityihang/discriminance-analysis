import torch

import pandas as pd
import torch.nn as nn
import torch.optim as optim


from models.Model import ClaModel

data = pd.read_csv('byh/src/datasets/winequality-red_test.csv')


device = torch.device("mps")

inputs= data.drop(columns='quality_two').values
outputs = data[["quality_two"]].values

inputs = torch.Tensor(inputs)
outputs = torch.Tensor(outputs)


model = ClaModel(11,32,1)

# 加载模型权重  
model_weights = torch.load("/Users/a11/vscode/discriminance-analysis/byh/checkpoint/Cla_model.pth")  
  
# 将加载的权重加载到模型实例中  
model.load_state_dict(model_weights) 
model.to(device)

model.eval()

# 将测试集转移到CPU设备上
test_inputs = inputs.to(device)
test_targets = outputs.to(device)

# 通过模型生成预测
predictions = model(test_inputs)


# 将预测转换为类别
predicted_classes = (predictions.squeeze() > 0.5).long()

# 计算准确率
num_correct = (predicted_classes == test_targets.squeeze()).sum().item()
num_total = test_targets.shape[0]
accuracy = num_correct/num_total


print(f"Test Accuracy: {accuracy:.4f}")   #Test Accuracy：0.8719