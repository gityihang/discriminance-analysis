#读取数据将其变成二分类的数据
import pandas as pd
from sklearn.model_selection import train_test_split 

data = pd.read_csv('data/winequality-red.csv')
#将红酒质量>6.5划分为好【用0表示】
data['quality_two'] = data['quality'].apply(lambda x: '0' if x > 6.5 else '1')
data.drop(columns='quality', axis=1, inplace=True)

data.to_csv('byh/src/datasets/winequality-red_binary.csv', index=False)

#将数据按照0.8比例分成训练集和测试集

train, test = train_test_split(data, test_size=0.2, random_state=42) 
train.to_csv('byh/src/datasets/winequality-red_train.csv', index=False)  
test.to_csv('byh/src/datasets/winequality-red_test.csv', index=False) 
