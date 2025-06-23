import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


import matplotlib.pyplot as plt
from tqdm import tqdm  # Jupyter 아닌 환경에서는 notebook 대신 이걸 사용

# 파일 경로를 직접 지정하거나 상대경로로 읽어오기
node_df = pd.read_csv("C:/Users/your_name/your_project_path/node_df.csv")
edge_df = pd.read_csv("C:/Users/your_name/your_project_path/edge_df.csv")

# 시각화 시 팝업으로 이미지가 뜨도록 설정 (VCS에서는 GUI로 뜸)
plt.figure(figsize=(8, 6))
plt.plot(node_df['timestamp'], node_df['value'])  # 예시
plt.title("Node Value Over Time")
plt.show()


NUM_NODES = node_df["node_id"].max() + 1
FEATURE_DIM = 3  # views, likes, comments
HIDDEN_DIM = 16  # Memory 차원 수


def safe_log(x):
    return torch.log1p(torch.tensor(x, dtype=torch.float32))

node_features = torch.zeros(NUM_NODES, FEATURE_DIM)
for _, row in node_df.iterrows():
    node_id = int(row["node_id"])
    node_features[node_id] = safe_log([row["views"], row["likes"], row["comments"]])


memory = torch.zeros(NUM_NODES, HIDDEN_DIM)
gru = nn.GRUCell(input_size=FEATURE_DIM, hidden_size=HIDDEN_DIM)


edge_df_sorted = edge_df.sort_values(by="timestamp")


for _, row in tqdm(edge_df_sorted.iterrows(), total=len(edge_df_sorted)):
    src = int(row["src"])
    dst = int(row["dst"])
    src_feat = node_features[src]
    prev_state = memory[dst]
    updated_state = gru(src_feat.unsqueeze(0), prev_state.unsqueeze(0)).squeeze(0)
    memory[dst] = updated_state


memory_sample = memory[:, :5].detach().numpy()

plt.figure(figsize=(12, 6))
sns.heatmap(memory_sample.T, cmap="viridis", cbar=True)
plt.title("Node Memory State (First 5 Dimensions per Node)")
plt.xlabel("Node ID")
plt.ylabel("Memory Dimension")
plt.tight_layout()
plt.show()
