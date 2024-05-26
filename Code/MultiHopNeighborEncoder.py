import torch
from torch.nn import Module, Linear, ReLU, Parameter
from torch_geometric.nn import GATConv
from torch import sigmoid


class MultiHopNeighborEncoder(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=2, dropout=0.6):
        super(MultiHopNeighborEncoder, self).__init__()

        assert in_channels > 0 and hidden_channels > 0 and out_channels > 0, "Channel dimensions must be positive."
        assert num_heads > 0, "Number of heads in GATConv must be positive."

        self.gat_1hop = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.linear_transform = Linear(hidden_channels * num_heads, out_channels)
        self.time_gate = Linear(out_channels * 2, 1)  # 用于时间感知的门控机制
        self.semantic_gate = Linear(out_channels * 2 + hidden_channels, 1)  # 调整以包含关系嵌入
        self.relu = ReLU()
        self.sigmoid = sigmoid()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout

        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self):
        self.gat_1hop.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear_transform.weight)
        torch.nn.init.zeros_(self.linear_transform.bias)
        torch.nn.init.xavier_uniform_(self.time_gate.weight)
        torch.nn.init.zeros_(self.time_gate.bias)
        torch.nn.init.xavier_uniform_(self.semantic_gate.weight)
        torch.nn.init.zeros_(self.semantic_gate.bias)

    def forward(self, x, edge_index_1hop, edge_time_1hop, edge_index_2hop, edge_time_2hop, rel_embeddings):
        # 检查输入尺寸
        assert x.size(
            1) == self.in_channels, f"Input feature size mismatch. Expected {self.in_channels}, got {x.size(1)}."
        assert edge_index_1hop.size(1) == edge_time_1hop.size(0), "Edge indices and times must match."
        assert edge_index_2hop.size(1) == edge_time_2hop.size(0), "Edge indices and times must match."

        x_1hop = self.aggregate_first_hop(x, edge_index_1hop, edge_time_1hop)
        x_fused = self.fuse_with_second_hop(x_1hop, edge_index_2hop, edge_time_2hop, rel_embeddings)

        return x_fused

    def aggregate_first_hop(self, x, edge_index, edge_time):
        """聚合一跳邻居信息"""
        x_1hop = self.gat_1hop(x, edge_index)
        x_1hop = self.linear_transform(x_1hop)
        time_emb_1hop = torch.unsqueeze(edge_time, dim=-1)
        x_1hop_with_time = torch.cat((x_1hop, time_emb_1hop), dim=-1)
        return x_1hop_with_time

    def fuse_with_second_hop(self, x_1hop_with_time, edge_index_2hop, edge_time_2hop, rel_embeddings):
        """融合一跳和二跳邻居信息"""
        # 获取一跳关系的嵌入
        rel_emb_1hop = rel_embeddings[edge_index_1hop[1]]
        # 调整语义门控输入以匹配线性层大小
        combined_emb_1hop = torch.cat((x_1hop_with_time[edge_index_1hop[0]], rel_emb_1hop, time_emb_1hop), dim=-1)
        semantic_weight = self.sigmoid(self.semantic_gate(combined_emb_1hop))

        # 聚合二跳邻居信息
        x_2hop = self.gather_two_hop_info(x, edge_index_2hop, semantic_weight, edge_time_2hop)
        x_2hop_with_time = torch.cat((x_2hop, time_emb_1hop), dim=-1)

        # 门控融合
        gate_input = torch.cat((x_1hop_with_time, x_2hop_with_time), dim=-1)
        gate_weights = self.sigmoid(self.time_gate(gate_input))
        x_fused = gate_weights * x_2hop_with_time + (1 - gate_weights) * x_1hop_with_time

        return x_fused

    def gather_two_hop_info(self, x, edge_index, semantic_weight, edge_time):
        """聚集二跳邻居信息，利用语义相关性进行加权"""
        x_2hop_unweighted = self.gat_1hop(x, edge_index)
        x_2hop_unweighted = self.linear_transform(x_2hop_unweighted)
        x_2hop = x_2hop_unweighted * semantic_weight.unsqueeze(-1)
        return torch.sum(x_2hop, dim=1)

    def aggregate_neighbors(self, task_rel, timestamps):
        """
        聚合多跳邻居信息以生成时间感知实体表示。
        假设timestamps是与task_rel相关的不同跳数的时间戳列表。
        """
        # 初始化邻居表示为空张量列表，用于存储每跳邻居的聚合表示
        neighbor_representations = []

        for hop in range(2):  # 假设最大两跳邻居
            # 获取对应跳数的邻居实体ID列表（此处需根据实际数据结构实现）
            hop_neighbors = self.get_hop_neighbors(task_rel, hop)

            # 获取邻居的初始嵌入
            neighbor_embeddings = self.entity_embeddings(hop_neighbors)

            # 将时间戳信息融入邻居表示，可以通过简单乘法实现
            # 这里假设timestamps是与邻居对应的，若非如此则需调整逻辑
            time_influenced_embeddings = neighbor_embeddings * timestamps[hop]

            # 对本跳邻居进行聚合，可以是求和、平均等，这里使用平均作为示例
            if len(hop_neighbors) > 0:  # 防止除以0
                aggregated_embedding = time_influenced_embeddings.mean(dim=0)
            else:  # 如果该跳没有邻居，则使用零向量
                aggregated_embedding = torch.zeros_like(neighbor_embeddings[0])

            neighbor_representations.append(aggregated_embedding)

        # 将所有跳数的邻居表示拼接起来形成最终的时间感知实体表示
        time_aware_entity_repr = torch.cat(neighbor_representations, dim=-1)
        return time_aware_entity_repr

    def positional_encoding(self, seq_len, d_model):
        '''位置编码'''
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

def example_usage():
    num_nodes = 100
    in_channels = 64
    hidden_channels = 32
    out_channels = 16
    num_relations = 20
    num_heads = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiHopNeighborEncoder(in_channels, hidden_channels, out_channels, num_heads).to(device)
    x = torch.randn((num_nodes, in_channels), device=device)
    edge_index_1hop = torch.randint(0, num_nodes, (2, num_nodes), device=device)
    edge_time_1hop = torch.rand(num_nodes, device=device)
    edge_index_2hop = torch.randint(0, num_nodes, (2, num_nodes), device=device)
    edge_time_2hop = torch.rand(num_nodes, device=device)
    rel_embeddings = torch.randn((num_relations, hidden_channels), device=device)

    output = model(x, edge_index_1hop, edge_time_1hop, edge_index_2hop, edge_time_2hop, rel_embeddings)
    print(f"Output shape after encoding: {output.shape}")


if __name__ == "__main__":
    example_usage()