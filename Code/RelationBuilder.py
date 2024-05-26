import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class RelationRepresentationBuilder(nn.Module):
    def __init__(self, embed_dim=200, num_heads=4, num_layers=2, dropout=0.1):
        """
        初始化关系生成器模块。

        参数:
        - embed_dim: 嵌入维度，决定实体和关系表示的大小。
        - num_heads: Transformer层中的注意力头数。
        - num_layers: Transformer编码器的层数。
        - dropout: Dropout比率，用于防止过拟合。
        """
        super(RelationRepresentationBuilder, self).__init__()
        # 定义Transformer编码器层
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=dropout),
            num_layers=num_layers
        )
        # Dropout层，用于正则化
        self.dropout = nn.Dropout(dropout)

        # 生成位置编码的函数
        self.positional_encoding = self._generate_sinusoidal_positional_encoding(embed_dim, max_len=1000)

    def _generate_sinusoidal_positional_encoding(self, d_model, max_len=1000):
        """
        生成正弦位置编码矩阵。

        参数:
        - d_model: 嵌入维度
        - max_len: 最大序列长度
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, time_aware_entity_pairs):
        """
        前向传播函数，用于生成关系表示。

        参数:
        - time_aware_entity_pairs: 形状为(K, embed_dim)的张量，其中K代表实体对的数量，embed_dim是嵌入维度。
        """
        # 添加位置编码
        seq_length = time_aware_entity_pairs.size(0)
        pos_enc = self.positional_encoding[:, :seq_length, :]
        time_aware_entity_pairs += pos_enc

        # Transformer编码
        # 需要增加一个批量维度，因为Transformer要求输入形状为(Batch, SeqLength, EmbeddingDim)
        transformer_output = self.transformer_encoder(time_aware_entity_pairs.unsqueeze(0))

        # 平均池化得到关系表示
        relation_representation = torch.mean(transformer_output.squeeze(0), dim=0)
        relation_representation = self.dropout(relation_representation)  # 应用Dropout

        return relation_representation

    # 假设参数
    embed_dim = 200
    num_heads = 8
    num_layers = 2
    dropout_rate = 0.1

    # 实例化模型
    relation_builder = RelationRepresentationBuilder(embed_dim, num_heads, num_layers, dropout=dropout_rate)

    # 示例数据，K个时间感知实体对
    K = 5  # 实体对数量
    example_time_aware_pairs = torch.randn(K, embed_dim)  # 生成随机数据作为示例

    # 调用模型前向传播
    relation_embedding = relation_builder(example_time_aware_pairs)

    print("Generated relation embedding shape:", relation_embedding.shape)