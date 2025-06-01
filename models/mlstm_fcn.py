import torch
import torch.nn as nn
import torch.nn.functional as F

# 为了独立运行，我们在这里硬编码这些值，实际应用中应从常量模块导入
MAX_TIMESTEPS = 640
MAX_NB_VARIABLES = 5
NB_CLASS = 9
LSTM_UNITS = 8 # Keras模型中的LSTM单元数
DROPOUT_RATE = 0.8 # Keras模型中的Dropout率

class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Keras Dense (filters // 16) -> PyTorch Linear (in_features, out_features)
        squeezed_channels = in_channels // reduction_ratio
        if squeezed_channels == 0: # 避免除以零或者通道数过小
            squeezed_channels = 1
        self.fc1 = nn.Linear(in_channels, squeezed_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(squeezed_channels, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels) # (batch, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1) # (batch, channels, 1)
        return x * y

class MLSTM_FCN(nn.Module):
    def __init__(self, max_nb_variables, max_timesteps, nb_class, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE):
        super(MLSTM_FCN, self).__init__()
        self.max_nb_variables = max_nb_variables
        self.max_timesteps = max_timesteps
        self.nb_class = nb_class

        # LSTM 路径
        # Keras Masking 层在此处未直接转换。PyTorch LSTM 可以处理打包序列以实现变长。
        # 此处假设输入已填充或长度固定。
        # Keras LSTM 输入 (batch, features, timesteps) -> permute -> (batch, timesteps, features) for PyTorch
        self.dropout_lstm = nn.Dropout(dropout_rate)

        self.lstm_corrected = nn.LSTM(input_size=self.max_nb_variables, # 特征数
                                         hidden_size=lstm_units,
                                         batch_first=True)


        # 卷积路径
        # Keras: y = Permute((2, 1))(ip) -> (batch, MAX_TIMESTEPS, MAX_NB_VARIABLES)
        # Conv1D(128, 8) -> in_channels = MAX_NB_VARIABLES
        # PyTorch Conv1D输入: (batch, MAX_NB_VARIABLES, MAX_TIMESTEPS)
        self.conv1 = nn.Conv1d(in_channels=self.max_nb_variables, out_channels=128, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.se1 = SqueezeExciteBlock(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        self.se2 = SqueezeExciteBlock(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1) # 对应 Keras GlobalAveragePooling1D

        # 全连接层
        self.fc_out = nn.Linear(lstm_units + 128, nb_class)

        # 初始化权重 (可选, PyTorch有默认的Kaiming初始化)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m.bias is not None: # Keras中use_bias=False
                 if m.bias is not None: # 检查以防万一
                    nn.init.constant_(m.bias, 0)


    def forward(self, ip):
        
        x_lstm_input = ip.permute(0, 2, 1)  # (batch_size, MAX_TIMESTEPS, MAX_NB_VARIABLES)
        x_lstm_out, _ = self.lstm_corrected(x_lstm_input)
        # Keras LSTM默认只返回最后一个时间步的输出
        x_lstm_processed = x_lstm_out[:, -1, :]  # (batch_size, lstm_units)
        x_lstm_processed = self.dropout_lstm(x_lstm_processed)

        y_conv = ip # (batch_size, MAX_NB_VARIABLES, MAX_TIMESTEPS)

        y_conv = self.conv1(y_conv)
        y_conv = self.bn1(y_conv)
        y_conv = F.relu(y_conv)
        y_conv = self.se1(y_conv)

        y_conv = self.conv2(y_conv)
        y_conv = self.bn2(y_conv)
        y_conv = F.relu(y_conv)
        y_conv = self.se2(y_conv)

        y_conv = self.conv3(y_conv)
        y_conv = self.bn3(y_conv)
        y_conv = F.relu(y_conv)
        y_conv = self.global_avg_pool(y_conv)  # (batch_size, 128, 1)
        y_conv = y_conv.squeeze(-1)  # (batch_size, 128)

        # 特征拼接
        x_combined = torch.cat((x_lstm_processed, y_conv), dim=1)

        # 输出层
        out = self.fc_out(x_combined)
        # Softmax通常在损失函数中处理 (例如 nn.CrossEntropyLoss)
        # 如果需要直接输出概率: out = F.softmax(out, dim=1)
        return out

# 示例用法:
if __name__ == '__main__':
    # 创建一个虚拟输入张量
    batch_size = 4
    # Keras Input shape (MAX_NB_VARIABLES, MAX_TIMESTEPS)
    # PyTorch input (batch_size, MAX_NB_VARIABLES, MAX_TIMESTEPS)
    dummy_input = torch.randn(batch_size, MAX_NB_VARIABLES, MAX_TIMESTEPS)

    # 实例化模型
    model_pytorch = MLSTM_FCN(max_nb_variables=MAX_NB_VARIABLES,
                                       max_timesteps=MAX_TIMESTEPS,
                                       nb_class=NB_CLASS)
    print(model_pytorch)

    # 执行前向传播
    output = model_pytorch(dummy_input)
    print("Output shape:", output.shape)  # 应该为 (batch_size, NB_CLASS)

    # 检查参数数量 (可选, 用于与Keras模型比较)
    total_params = sum(p.numel() for p in model_pytorch.parameters() if p.requires_grad)
    print(f"Total trainable parameters (PyTorch): {total_params}")
