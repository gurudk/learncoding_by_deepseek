import torch
import torch.nn as nn

# 配置参数
input_size = 1  # 输入特征维度
hidden_size = 16  # 隐藏层维度
output_size = 1  # 输出维度
seq_len = 5  # 序列长度

# 生成训练数据：输入[0,1,2,3,4]，目标输出[1,2,3,4,5]
x = torch.FloatTensor([i for i in range(seq_len)]).view(seq_len, 1, 1)
y = torch.FloatTensor([i + 1 for i in range(seq_len)]).view(seq_len, 1, 1)


# 定义RNN模型
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=False
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态（num_layers, batch_size, hidden_size）
        h0 = torch.zeros(1, x.size(1), hidden_size)

        # RNN输出：seq_len, batch_size, hidden_size
        out, _ = self.rnn(x, h0)

        # 全连接层处理每个时间步的输出
        out = self.fc(out)
        return out


# 初始化模型、损失函数和优化器
model = SimpleRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# 测试预测
with torch.no_grad():
    test_input = torch.FloatTensor([[5]]).view(1, 1, 1)
    predicted = model(test_input)
    print(f'Input: 5 => Predicted: {predicted.item():.1f}')