import torch
import torch.nn as nn
from torch.nn import init


class EnhancedKGEmbeddings(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim, time_emb_dim=32, use_time_decay=True):
        super(EnhancedKGEmbeddings, self).__init__()

        self.entity_embeddings = nn.Embedding(num_entities, emb_dim)
        self.relation_embeddings = nn.Embedding(num_relations, emb_dim)

        # 新增时间嵌入层
        self.time_embeddings = nn.Embedding(num_relations, time_emb_dim)  # 假设每个关系类型有独立的时间嵌入

        # 可选：使用时间衰减因子
        self.use_time_decay = use_time_decay
        if use_time_decay:
            self.time_decay = nn.Parameter(torch.Tensor(1))  # 可学习的时间衰减系数

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.entity_embeddings.weight)
        init.xavier_uniform_(self.relation_embeddings.weight)
        init.xavier_uniform_(self.time_embeddings.weight)

        # 初始化时间衰减参数
        if self.use_time_decay:
            init.uniform_(self.time_decay, -1.0, 1.0)

    def forward(self, entity_indices, relation_indices, timestamp_indices=None):
        """
        entity_indices: [batch_size] 实体索引
        relation_indices: [batch_size] 关系索引
        timestamp_indices: [batch_size] 时间戳索引（可选）
        """
        entity_embs = self.entity_embeddings(entity_indices)
        relation_embs = self.relation_embeddings(relation_indices)

        if timestamp_indices is not None:
            # 将时间戳嵌入与关系嵌入相结合
            time_embs = self.time_embeddings(relation_indices)  # 假设时间嵌入与关系类型绑定
            if self.use_time_decay:
                # 应用时间衰减
                decayed_time_embs = time_embs * torch.exp(-self.time_decay * timestamp_indices.float().unsqueeze(-1))
                combined_embs = relation_embs + decayed_time_embs
            else:
                combined_embs = torch.cat((relation_embs, time_embs), dim=-1)  # 简单拼接
        else:
            combined_embs = relation_embs

        # 最终输出可以是实体和结合时间信息的关系嵌入的某种组合，这里简化处理为直接返回
        return entity_embs, combined_embs