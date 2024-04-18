判别分析作业
红酒数据：一共有11个因变量，一个自变量，下面为数据的格式
fixed acidity           
volatile acidity        
citric acid             
residual sugar          
chlorides               
free sulfur dioxide     
total sulfur dioxide    
density                 
pH                      
sulphates               
alcohol                 
quality         
对于上述数据选取quality为自变量，采用判别分析对以上数据进行判别
由于quality是一个程度数据，现在规定当qulity大于6.5，则表示该红酒品质为好，标记为0        
                  
文档格式示例  ：
our_project/  
    data/            # 存放数据集和预处理后的数据  
    docs/            # 存放项目文档  
    models/          # 存放模型定义文件  
    utils/           # 存放实用函数和工具  
    src/  
        datasets/    # 数据集处理代码  
        models/      # 神经网络模型定义  
        training/    # 训练代码  
        evaluation/  # 评估代码  
        main.py      # 主程序入口  
    requirements.txt # 项目依赖列表  
    README.md        # 项目说明文档