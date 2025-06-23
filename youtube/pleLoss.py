import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. 데이터 로드
node_df = pd.read_csv("node_df.csv")
edge_df = pd.read_csv("edge_df.csv")

# 노드 id 매핑
node_id_to_idx = {nid: i for i, nid in enumerate(node_df['node_id'])}
num_nodes = len(node_id_to_idx)

# 노드 피처 추출 및 정규화
features = node_df[['views', 'likes', 'comments']].values.astype(np.float32)
min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
node_features = (features - min_vals) / (max_vals - min_vals + 1e-6)

# 피처 차원 확장
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
    deg = adj.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1
    adj = adj / deg
    return adj

# Edge Probability 예측 모듈
class EdgePredictor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, src_emb, dst_emb):
        x = torch.cat([src_emb, dst_emb], dim=1)
        return self.mlp(x).squeeze(1)

# 전체 모델 통합
class IntegratedModel(nn.Module):
    def __init__(self, num_nodes, memory_dim, heads=4, scn_dim=8):
        super().__init__()
        self.tgn_memory = TGNMemory(num_nodes, memory_dim)
        self.gt = GraphTransformer(memory_dim, heads)
        self.scn = SCNLayer(memory_dim, scn_dim)
        self.edge_predictor = EdgePredictor(scn_dim)
    def forward(self, src, dst, t, node_features, adj):
        self.tgn_memory.update_state(src, dst, t, node_features)
        mem = self.tgn_memory.memory.data
        gt_out, _ = self.gt(mem)
        scn_out = self.scn(gt_out, adj)
        src_emb = scn_out[src]
        dst_emb = scn_out[dst]
        edge_prob = self.edge_predictor(src_emb, dst_emb)
        return edge_prob

# 학습 루프
if __name__ == "__main__":
    torch.manual_seed(42)
    model = IntegratedModel(num_nodes, memory_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # 데이터 준비: 실제 엣지(양성) + 랜덤 엣지(음성)
    pos_edges = edge_df[['src', 'dst']].values
    pos_labels = [1] * len(pos_edges)
    neg_edges = []
    while len(neg_edges) < len(pos_edges):
        u, v = np.random.randint(0, num_nodes, 2)
        if not ((edge_df['src'] == u) & (edge_df['dst'] == v)).any() and u != v:
            neg_edges.append([u, v])
    neg_labels = [0] * len(neg_edges)

    all_edges = np.vstack([pos_edges, neg_edges])
    all_labels = np.array(pos_labels + neg_labels)

    # 데이터 셔플
    idx = np.arange(len(all_edges))
    np.random.shuffle(idx)
    all_edges, all_labels = all_edges[idx], all_labels[idx]

    # 인접 행렬 생성
    adj = create_adj_matrix(edge_df, num_nodes)

    # 학습 파라미터
    batch_size = 128
    num_epochs = 100 # 50->100->200->100

    for epoch in range(num_epochs):
        total_loss = 0
        all_preds = []  # 전체 예측값 저장
        all_true = []   # 전체 실제값 저장
        
        # 배치 단위 처리
        for i in range(0, len(all_edges), batch_size):
            # 배치 데이터 준비
            batch_edges = all_edges[i:i+batch_size]
            batch_labels = all_labels[i:i+batch_size]
            src = torch.tensor(batch_edges[:,0], dtype=torch.long)
            dst = torch.tensor(batch_edges[:,1], dtype=torch.long)
            labels = torch.tensor(batch_labels, dtype=torch.float32)
            
            # 모델 예측
            pred = model(src, dst, 0, node_features, adj)
            loss = criterion(pred, labels)
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item() * len(batch_edges)
            
            # 예측값 및 실제값 저장 (평가지표 계산용)
            all_preds.extend(torch.sigmoid(pred).detach().numpy())
            all_true.extend(labels.numpy())
        
        # 에폭별 평균 손실 계산
        avg_loss = total_loss / len(all_edges)
        
        # 평가지표 계산
        pred_binary = (np.array(all_preds) > 0.5).astype(int)
        precision = precision_score(all_true, pred_binary, zero_division=0)
        recall = recall_score(all_true, pred_binary, zero_division=0)
        f1 = f1_score(all_true, pred_binary, zero_division=0)
        
        # 결과 출력
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    print("학습 완료")

    torch.save(model.state_dict(), "trained_model.pth")
    print("모델 파라미터가 trained_model.pth로 저장되었습니다.")