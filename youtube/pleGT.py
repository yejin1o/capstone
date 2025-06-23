import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# 데이터 로드
node_df = pd.read_csv("node_df.csv")
edge_df = pd.read_csv("edge_df.csv")

# 노드 매핑 및 피처 정규화
node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_df['node_id'])}
num_nodes = len(node_id_to_idx)

# 노드 피처 추출 및 정규화
features = node_df[['views', 'likes', 'comments']].values.astype(np.float32)
min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
node_features = (features - min_vals) / (max_vals - min_vals + 1e-6)
node_features = torch.tensor(node_features, dtype=torch.float32)

# TGN 메모리 모듈
class TGNMemory(nn.Module):
    def __init__(self, num_nodes, memory_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim), requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)
        self.gru = nn.GRUCell(memory_dim, memory_dim)
    
    def update_state(self, src, dst, t, node_features):
        unique_nodes = torch.unique(torch.cat([src, dst]))
        new_memory = self.memory.data.clone()
        
        for node in unique_nodes:
            # 현재 노드와 연결된 모든 엣지 찾기
            mask = (src == node) | (dst == node)
            connected_nodes = torch.where(src == node, dst, src)[mask]
            
            if len(connected_nodes) > 0:
                # 이웃 노드 피처 집계 (평균)
                neighbor_features = node_features[connected_nodes]
                agg_msg = neighbor_features.mean(dim=0, keepdim=True)
                
                # GRU를 통한 메모리 업데이트
                prev_mem = self.memory[node].unsqueeze(0)
                updated_mem = self.gru(agg_msg, prev_mem)
                new_memory[node] = updated_mem.squeeze(0)
        
        # 메모리 및 타임스탬프 갱신
        self.memory.data.copy_(new_memory)
        self.last_update[unique_nodes] = t

# 메인 처리 함수
def process_temporal_graph(edge_df, node_features):
    # 타임스탬프별 엣지 그룹화
    timestamp_groups = edge_df.groupby('timestamp')
    memory_dim = node_features.shape[1]
    tgn_memory = TGNMemory(num_nodes, memory_dim)
    
    for timestamp, group in timestamp_groups:
        # 엣지 데이터를 텐서로 변환
        src_nodes = torch.tensor([node_id_to_idx[x] for x in group['src']], dtype=torch.long)
        dst_nodes = torch.tensor([node_id_to_idx[x] for x in group['dst']], dtype=torch.long)
        
        # 메모리 업데이트
        tgn_memory.update_state(src_nodes, dst_nodes, timestamp, node_features)
    
    return tgn_memory.memory.data

# 실행
if __name__ == "__main__":
    updated_memory = process_temporal_graph(edge_df, node_features)
    print("최종 메모리 상태:")
    print(updated_memory)
