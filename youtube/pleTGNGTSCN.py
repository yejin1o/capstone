import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# 1. 데이터 로드
node_df = pd.read_csv("node_df.csv")
edge_df = pd.read_csv("edge_df.csv")

# 노드 id 매핑
node_id_to_idx = {nid: i for i, nid in enumerate(node_df['node_id'])}
num_nodes = len(node_id_to_idx)

# 노드 피처 추출 및 정규화 (3차원: views, likes, comments)
features = node_df[['views', 'likes', 'comments']].values.astype(np.float32)
min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
node_features = (features - min_vals) / (max_vals - min_vals + 1e-6)

# 피처 차원 확장 (3차원 -> 8차원)
memory_dim = 8
if node_features.shape[1] < memory_dim:
    pad_width = memory_dim - node_features.shape[1]
    node_features = np.pad(node_features, ((0, 0), (0, pad_width)), 'constant')
node_features = torch.tensor(node_features, dtype=torch.float32)

# TGN Memory 모듈
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
            mask = (src == node) | (dst == node)
            connected_nodes = torch.where(src == node, dst, src)[mask]
            if len(connected_nodes) > 0:
                neighbor_features = node_features[connected_nodes]
                agg_msg = neighbor_features.mean(dim=0, keepdim=True)
                prev_mem = self.memory[node].unsqueeze(0)
                updated_mem = self.gru(agg_msg, prev_mem)
                new_memory[node] = updated_mem.squeeze(0)
        self.memory.data.copy_(new_memory)
        self.last_update[unique_nodes] = t

# GT(Graph Transformer) 모듈
class GraphTransformer(nn.Module):
    def __init__(self, dim, heads=4, ffn_dim=64):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x, adj_mask=None):
        x = x.unsqueeze(0)
        attn_out, attn_weights = self.attn(x, x, x, attn_mask=adj_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.squeeze(0), attn_weights.squeeze(0)

# SCN(Subgraph Convolution Network) 모듈
class SCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj):
        return self.linear(torch.matmul(adj, x))

# 인접 행렬 생성 함수
def create_adj_matrix(edge_df, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for _, row in edge_df.iterrows():
        src = node_id_to_idx[row['src']]
        dst = node_id_to_idx[row['dst']]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    # 정규화
    deg = adj.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1
    adj = adj / deg
    return adj

if __name__ == "__main__":
    memory_dim = 8
    scn_dim = 8
    heads = 4

    # 1. TGN 메모리 업데이트 (전체 엣지 시간 순서대로)
    tgn_memory = TGNMemory(num_nodes, memory_dim)
    timestamp_groups = edge_df.groupby('timestamp')
    for timestamp, group in timestamp_groups:
        src_nodes = torch.tensor([node_id_to_idx[x] for x in group['src']], dtype=torch.long)
        dst_nodes = torch.tensor([node_id_to_idx[x] for x in group['dst']], dtype=torch.long)
        tgn_memory.update_state(src_nodes, dst_nodes, timestamp, node_features)

    # 2. GT 임베딩 생성
    adj_mask = None  # 필요시 create_adj_matrix(edge_df, num_nodes) > 0로 mask 생성
    gt = GraphTransformer(memory_dim, heads)
    gt_emb, attn_weights = gt(tgn_memory.memory.data, adj_mask)

    # 3. SCN 임베딩 생성
    adj = create_adj_matrix(edge_df, num_nodes)
    scn = SCNLayer(memory_dim, scn_dim)
    scn_emb = scn(gt_emb, adj)

    # 4. 결과 저장 (CSV)
    result_df = pd.DataFrame({
        'node_id': node_df['node_id'],
        'tgn_memory': [arr.detach().numpy() for arr in tgn_memory.memory.data],
        'gt_embedding': [arr.detach().numpy() for arr in gt_emb],
        'scn_embedding': [arr.detach().numpy() for arr in scn_emb]
    })
    result_df.to_csv("tgn_gt_scn_results.csv", index=False)
    print("TGN→GT→SCN 임베딩 생성 및 저장 완료.")
