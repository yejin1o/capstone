import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------------
# 데이터 로딩 및 전처리
# -------------------------------

# 데이터 경로 (수정 필요)
node_df = pd.read_csv("C:/Users/AI-KYJ/Desktop/2025/capstone/Code/node_df.csv")
edge_df = pd.read_csv("C:/Users/AI-KYJ/Desktop/2025/capstone/Code/edge_df.csv")

# 시간 순 정렬
edge_df_sorted = edge_df.sort_values(by="timestamp")

# 설정값
NUM_NODES = node_df["node_id"].max() + 1
FEATURE_DIM = 3  # views, likes, comments
HIDDEN_DIM = 16

# 노드 피처 생성
node_features = torch.zeros(NUM_NODES, FEATURE_DIM)
for _, row in node_df.iterrows():
    node_id = int(row["node_id"])
    node_features[node_id] = torch.log1p(torch.tensor([
        row["views"], row["likes"], row["comments"]
    ], dtype=torch.float32))

# 예측할 edge 후보 (예시)
edge_candidates = [(1, 2), (3, 4), (5, 10)]

# -------------------------------
# GT 기반 TGN 모델 정의
# -------------------------------

class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x

class GraphTransformer(nn.Module):
    def __init__(self, dim, heads, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(dim, heads) for _ in range(layers)
        ])

    def forward(self, node_embeddings):
        x = node_embeddings.unsqueeze(0)  # (1, N, D)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(0)  # (N, D)

class SCN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, node_embs):
        return self.conv(node_embs)

class EdgePredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, src_emb, dst_emb):
        h = torch.cat([src_emb, dst_emb], dim=-1)
        return torch.sigmoid(self.predictor(h))

class GT_TGN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 학습 파라미터가 아님! requires_grad = False
        self.register_buffer("memory", torch.zeros(num_nodes, hidden_dim))

        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.gt = GraphTransformer(hidden_dim, heads)
        self.scn = SCN(hidden_dim)
        self.edge_predictor = EdgePredictor(hidden_dim)

    def update_memory(self, src_feat, dst_id):
        prev = self.memory[dst_id]
        updated = self.gru(src_feat.unsqueeze(0), prev.unsqueeze(0)).squeeze(0)
        # 역전파 깨짐 방지를 위해 복사
        self.memory[dst_id] = updated.detach().clone()

    def forward(self, node_features, edge_df_sorted, edge_candidates):
        for _, row in tqdm(edge_df_sorted.iterrows(), total=len(edge_df_sorted)):
            src = int(row["src"])
            dst = int(row["dst"])
            src_feat = node_features[src]
            self.update_memory(src_feat, dst)

        node_embeddings = self.gt(self.memory)
        refined_embeddings = self.scn(node_embeddings)

        preds = []
        for src, dst in edge_candidates:
            prob = self.edge_predictor(refined_embeddings[src], refined_embeddings[dst])
            preds.append(prob)

        return torch.stack(preds).squeeze()

    def get_memory(self):
        return self.memory.detach()

# -------------------------------
# 모델 실행
# -------------------------------

model = GT_TGN(NUM_NODES, FEATURE_DIM, HIDDEN_DIM)
pred_probs = model(node_features, edge_df_sorted, edge_candidates)

print("Edge 확산 확률 예측 결과:", pred_probs.tolist())

# -------------------------------
# 데이터셋 분할 및 학습용 샘플 생성
# -------------------------------
import random
from sklearn.model_selection import train_test_split

# timestamp 기준 정렬된 edge_df 활용
cutoff = edge_df_sorted["timestamp"].quantile(0.8)
train_edges = edge_df_sorted[edge_df_sorted["timestamp"] <= cutoff]
test_edges = edge_df_sorted[edge_df_sorted["timestamp"] > cutoff]

# 학습용 양성 샘플
train_pos = [(int(r["src"]), int(r["dst"])) for _, r in train_edges.iterrows()]
train_pos_labels = [1] * len(train_pos)

# 학습용 음성 샘플 (negative sampling)
train_neg = []
while len(train_neg) < len(train_pos):
    src = random.randint(0, NUM_NODES - 1)
    dst = random.randint(0, NUM_NODES - 1)
    if not ((train_edges["src"] == src) & (train_edges["dst"] == dst)).any():
        train_neg.append((src, dst))
train_neg_labels = [0] * len(train_neg)

# 학습용 데이터 병합
train_candidates = train_pos + train_neg
train_labels = train_pos_labels + train_neg_labels

# 테스트셋 (동일한 방식, 평가용)
test_pos = [(int(r["src"]), int(r["dst"])) for _, r in test_edges.iterrows()]
test_pos_labels = [1] * len(test_pos)

test_neg = []
while len(test_neg) < len(test_pos):
    src = random.randint(0, NUM_NODES - 1)
    dst = random.randint(0, NUM_NODES - 1)
    if not ((test_edges["src"] == src) & (test_edges["dst"] == dst)).any():
        test_neg.append((src, dst))
test_neg_labels = [0] * len(test_neg)

test_candidates = test_pos + test_neg
test_labels = test_pos_labels + test_neg_labels

# -------------------------------
# 학습 함수
# -------------------------------
import torch.optim as optim
import torch.nn as nn

def train(model, node_features, edge_df_sorted, edge_candidates, edge_labels, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(node_features, edge_df_sorted, edge_candidates)
        labels = torch.tensor(edge_labels, dtype=torch.float32)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# -------------------------------
# 평가 함수
# -------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, node_features, edge_df_sorted, edge_candidates, true_labels):
    model.eval()
    with torch.no_grad():
        preds = model(node_features, edge_df_sorted, edge_candidates)
        pred_labels = (preds > 0.5).float().cpu().numpy()
        true_labels = torch.tensor(true_labels).cpu().numpy()
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels)
        rec = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# -------------------------------
# 학습 및 평가 실행
# -------------------------------
model = GT_TGN(NUM_NODES, FEATURE_DIM, HIDDEN_DIM)
train(model, node_features, edge_df_sorted, train_candidates, train_labels, epochs=100)
evaluate(model, node_features, edge_df_sorted, test_candidates, test_labels)
