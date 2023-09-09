## 功能
- 支持多种语义分割网络模型,包括UNet、R2UNet、Attention UNet、Nested UNet等
- 支持多种损失函数,包括交叉熵、Dice损失、结合二者的损失等
- 支持肺部CT数据集训练
- 包含训练和验证过程,可视化训练曲线
- 支持TensorBoard日志记录训练过程
- 支持模型断点续训
- 计算语义分割评价指标并可视化
- 支持Multi-GPU训练

## 用法
`python train.py` 

主要参数:

- `--model`:选择模型,默认UNet
- `--loss`:选择损失函数,默认Dice损失
- `--dataset`:选择数据集,默认肺部CT
- `--epoch`:训练epoch数,默认300
- `--batch_size`:批大小,默认2
- `--lr`:学习率,默认1e-4
- `--checkpoint`:加载预训练模型
- `--gpu`:使用的GPU设备
- `--parallel`:是否多GPU训练

程序会自动记录TensorBoard日志,保存模型断点。

## 代码说明

- `models/` - 不同模型的网络结构
- `datasets/` - 数据集加载器
- `utils/` - 评价指标计算等函数

主要流程:

1. 设置参数,读取数据
2. 构建模型,定义优化器和损失函数
3. 训练循环:前向传播、反向传播、优化
4. 计算指标,记录TensorBoard
5. 保存模型断点

可扩展新增模型及数据集,提供了完整的训练验证框架。