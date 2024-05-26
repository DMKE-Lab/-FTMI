import pandas as pd
from collections import defaultdict, namedtuple
from torch.utils.data import Dataset

# 定义一个四元组命名元组以更好地表示时态知识图谱的数据
KGQuadruple = namedtuple('KGQuadruple', ['head', 'relation', 'tail', 'timestamp'])

class TemporalKGDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.data = pd.read_csv(file_path)
        self.entities, self.relations, self.timestamps = self._extract_entities_relations_timestamps()
        self.entity_to_idx, self.idx_to_entity = self._index_entities()
        self.relation_to_idx, self.idx_to_relation = self._index_relations()
        self.timestamp_to_idx, self.idx_to_timestamp = self._index_timestamps()  # 新增时间戳索引
        self.quadruples = self._construct_quadruples()  # 修改为处理四元组
        self.neighbors = self._build_temporal_neighbor_dict()  # 修改以包含时间信息

    def _extract_entities_relations_timestamps(self):
        entities = set(self.data['head']) | set(self.data['tail'])
        relations = set(self.data['relation'])
        timestamps = set(self.data['timestamp'])
        return entities, relations, timestamps

    def _index_timestamps(self):
        timestamp_to_idx = {ts: i for i, ts in enumerate(sorted(self.timestamps))}
        idx_to_timestamp = {i: ts for ts, i in timestamp_to_idx.items()}
        return timestamp_to_idx, idx_to_timestamp

    def _construct_quadruples(self):
        quadruples = [KGQuadruple(
            self.entity_to_idx[h], self.relation_to_idx[r], self.entity_to_idx[t], self.timestamp_to_idx[ts])
            for h, r, t, ts in zip(
                self.data['head'], self.data['relation'], self.data['tail'], self.data['timestamp'])
        ]
        return quadruples

    def _build_temporal_neighbor_dict(self):
        neighbors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # 添加时间维度
        for h, r, t, ts in self.quadruples:
            neighbors[h][r][ts].append(t)  # 添加时间戳信息
            neighbors[t][r + max(self.relation_to_idx.values())][ts].append(h)  # 反向关系同样记录时间戳
        return neighbors

    def __getitem__(self, index):
        """重写以支持四元组数据获取"""
        quadruple = self.quadruples[index]
        return (
            quadruple.head, quadruple.relation, quadruple.tail,
            quadruple.timestamp, self.idx_to_entity[quadruple.head],
            self.idx_to_relation[quadruple.relation], self.idx_to_entity[quadruple.tail],
            self.idx_to_timestamp[quadruple.timestamp]
        )

    def __len__(self):
        return len(self.quadruples)