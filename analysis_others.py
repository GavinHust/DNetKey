import random
import numpy as np
import networkx as nx
import os
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from networkx.algorithms import centrality
from torch_geometric.data import Data
from collections import deque


class ResidualGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.05):  # dropout=0.6
        super(ResidualGATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = lambda x: x

    def forward(self, x, edge_index):
        res = self.residual(x)
        x = self.gat(x, edge_index)
        x = self.norm(x)
        x = self.dropout(x)
        return x + res


class DeepGATNet(nn.Module):
    def __init__(self, in_features, hidden_dims, out_features, heads_per_layer, mlp_dims):
        super(DeepGATNet, self).__init__()
        assert len(hidden_dims) == len(heads_per_layer), "Hidden dimensions and heads per layer counts must match."
        self.layers = nn.ModuleList()

        # 添加GAT层
        current_dim = in_features
        for dim, heads in zip(hidden_dims, heads_per_layer):
            self.layers.append(ResidualGATLayer(current_dim, dim, heads))
            current_dim = dim

        # 添加最后一层GATConv，不使用残差连接
        self.layers.append(GATConv(current_dim, out_features, heads=1, concat=False))

        # 添加全连接层（多层感知器）
        self.mlp = nn.Sequential(
            nn.Linear(out_features, mlp_dims[0]),
            nn.ELU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ELU(),
            nn.Linear(mlp_dims[1], mlp_dims[2]),
            nn.Sigmoid()  # 最后一层使用Sigmoid激活函数
        )

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:  # 前面的GAT层
            x = layer(x, edge_index)
        x = self.layers[-1](x, edge_index)  # 最后一层GAT
        x = self.mlp(x)
        return x.squeeze()


def load_DND(model, optimizer, filepath):
    print(device)
    checkpoint = torch.load(filepath, map_location=device)
    # print(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()  # 切换到评估模式
    return model, optimizer


def DND_features(G):
    features = {
        "in_degree": [degree for node, degree in G.in_degree()],
        "out_degree": [degree for node, degree in G.out_degree()],
        "betweenness": list(centrality.betweenness_centrality(G).values()),
        "pagerank": list(nx.pagerank(G).values()),
    }
    features_dict = features
    features = np.array([features_dict[key] for key in sorted(features_dict.keys())]).T

    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
    edge_index = np.array(adj_matrix.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float)  # 使用计算的特征
    # print(x)
    # print(edge_index)
    # print(x.shape)
    # print(edge_index.shape)
    return Data(x=x, edge_index=edge_index)


def corehd_disintegration(G):
    disintegrate_order = []
    nodes = list(G.nodes())  # 获取图的节点列表

    A = np.array(nx.adjacency_matrix(G).todense(), dtype=float)  # 原始图的邻接矩阵

    # 开始 2-core 分解
    while True:
        # 提取当前 2-core 子图
        two_core = nx.k_core(G, k=2)
        if two_core.number_of_nodes() == 0:
            break
        two_core_nodes = list(two_core.nodes())
        A_two_core = np.array(nx.adjacency_matrix(two_core).todense(), dtype=float)
        D_two_core = A_two_core.sum(axis=0) + A_two_core.sum(axis=1)
        d = np.argmax(D_two_core)
        node_to_remove = two_core_nodes[d]
        disintegrate_order.append(node_to_remove)
        G.remove_node(node_to_remove)
        for i, x in enumerate(nodes):
            if x == node_to_remove:
                indices = i
        A[:, indices] = 0
        A[indices, :] = 0

    # 处理剩余节点
    remaining_nodes = [x for x in nodes if x not in disintegrate_order]
    # print(f"Length of 2-core disintegration order: {len(disintegrate_order)}")
    # print(f"Remaining nodes: {len(remaining_nodes)}")

    # 使用广度优先搜索对剩余节点进行处理
    if remaining_nodes:
        visited = set()
        queue = deque()

        # 处理每一个剩余节点
        for node in remaining_nodes:
            node_idx = nodes.index(node)  # 查找原始节点的索引
            if node_idx not in visited:
                queue.append(node_idx)
                visited.add(node_idx)

                while queue:
                    current_node_idx = queue.popleft()
                    disintegrate_order.append(nodes[current_node_idx])

                    # 查找邻居节点并加入队列
                    for neighbor_idx in np.where(A[current_node_idx] != 0)[0]:
                        if neighbor_idx not in visited:
                            visited.add(neighbor_idx)
                            queue.append(neighbor_idx)

                    # 移除当前节点的所有连接
                    A[:, current_node_idx] = 0
                    A[current_node_idx, :] = 0

    # print(f"Total disintegration order length: {len(disintegrate_order)}")
    return disintegrate_order


def findfile(directory, file_prefix):
    filenames = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dir = "SF_ER_FINDER/ER/"          #SF网络数据
file_pre = "ER_1000_"

filenames = findfile(dir, file_pre)

epoch = 0
for file in filenames:
    print('epoch:', epoch)
    print(file)
    filename = file  # [:-4]
    g = nx.read_graphml(dir + filename)  # 读取graphhml形式储存的图
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边

    nodes = list(g.nodes)
    print(nodes)
    print(np.array(nodes))

    N = len(nodes)
    print("N",N)
    back_nodes = []
    selected = []

# 随机方法选取节点：                                                 #将节点顺序随机打乱
    print("rand")
    nodes_rand = list(g.nodes)
    random.shuffle(nodes_rand)

# 适应度方法选取节点：                                                #按照节点入度与出度之和从大到小删除，每次删除一个节点就重新计算各节点的入度与出度之和
    print("HDA")
    adapt_degree=[]
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)

    nodes=list(g.nodes())
    while A.sum():
        D = A.sum(axis=0) + A.sum(axis=1)
        d=np.argmax(D)
        adapt_degree.append(nodes[d])
        A[:, d] = 0  # 以前是行置零
        A[d,:] = 0
    adapt_degree = adapt_degree + list(set(g.nodes()) - set(adapt_degree))


# 度方法选取节点                                                   #按照节点最初始的入度与出度之和从大到小删除
    print("HD")
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
    D=A.sum(axis=0)+A.sum(axis=1)
    d = sorted(range(N), key=lambda k: D[k], reverse=True)
    #print(d)
    degree=np.array(list(g.nodes))[d]
    degree=degree.tolist()+list(set(g.nodes())-set(degree))

# 入度方法选取节点                                                   #按照节点最初始的入度与出度之和从大到小删除
    print("ID")
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
    D = A.sum(axis=0)
    d = sorted(range(N), key=lambda k: D[k], reverse=True)
    # print(d)
    degree_in = np.array(list(g.nodes))[d]
    degree_in = degree_in.tolist() + list(set(g.nodes()) - set(degree_in))


# 出度方法选取节点                                                   #按照节点最初始的入度与出度之和从大到小删除
    print("OD")
    A = np.array(nx.adjacency_matrix(g).todense(), dtype=float)
    D = A.sum(axis=1)
    d = sorted(range(N), key=lambda k: D[k], reverse=True)
    # print(d)
    degree_out = np.array(list(g.nodes))[d]
    degree_out = degree_out.tolist() + list(set(g.nodes()) - set(degree_out))


# MinSum方法取节点
    print("MiniSum")
    idx1 = np.load('SM_selected/' + file+'-output.npy').astype(int)
    print(idx1)
    #idx1 = np.loadtxt('SM_selected/' + file + '.npy').astype(int)
    #print(idx2)
    minisum_nodes = np.array(list(g.nodes))[idx1]
    minisum_nodes = minisum_nodes.tolist() + list(set(g.nodes()) - set(minisum_nodes))
    #print(minisum_nodes)


# PageRank方法取节点
    # 按照PageRank的值进行ND,每次删除重新计算PageRank，还是应该计算第一次的PageRank排序
    print("PageRank")
    pagerank_nodes=[]
    page_g = g.copy()
    nodes_d = len(list(page_g.nodes()))
    nodes = list(g.nodes())
    while nodes_d:
        pagerank_scores = nx.pagerank(page_g)
        #print(pagerank_scores)
        max_key = max(pagerank_scores, key=lambda k: pagerank_scores[k])
        #print(max_key)
        pagerank_nodes.append(max_key)
        page_g.remove_node(max_key)
        nodes_d = len(list(page_g.nodes()))
    pagerank_nodes = pagerank_nodes + list(set(g.nodes()) - set(pagerank_nodes))


    # DND方法取节点
    print("DND")
    mlp_dims = [100, 50, 1]  # 全连接层的神经元数量
    model_DND = DeepGATNet(in_features=4, hidden_dims=[40, 30, 20, 10], out_features=mlp_dims[0],
                           heads_per_layer=[5, 5, 5, 5], mlp_dims=mlp_dims).to(device)
    optimizer_DND = torch.optim.Adam(model_DND.parameters(), lr=0.000085)
    model_DND, optimizer_DND = load_DND(model_DND, optimizer_DND,'model_checkpoint_DND.pth')
    features_DND = DND_features(g)
    with torch.no_grad():
        model_DND.eval()
        out = model_DND(features_DND.x.to(device), features_DND.edge_index.to(device))
        # print("Test output:", out)
        # print(out.shape)
    sorted_indices_DND = torch.argsort(out, descending=True)  # 获取out节点标签预测值从大到小排列的索引值
    #print(sorted_indices_DND)
    DND_nodes = np.array(list(g.nodes))[sorted_indices_DND.tolist()]
    #DND_nodes = sorted_indices_DND.tolist()
    #print(DND_nodes)



# CoreHD方法取点
    print("CoreHD")
    corehd_g = g.copy()
    CoreHD_nodes  = corehd_disintegration(corehd_g)
    #print(CoreHD_nodes)


    g1 = g.copy()
    g2 = g.copy()
    g3 = g.copy()
    g4 = g.copy()
    g5 = g.copy()
    g6 = g.copy()
    g7 = g.copy()
    g8 = g.copy()
    g9 = g.copy()


    strong_list_rand = []
    strong_list_degree = []
    strong_list_degree_in = []
    strong_list_degree_out = []
    strong_list_adapt_degree = []
    strong_list_MS = []
    strong_list_pagerank = []
    strong_list_DND = []
    strong_list_Core = []


    for i in range(len(degree_in)):

        strong_rand = max(nx.strongly_connected_components(g2),key=len)
        strong_list_rand.append(len(strong_rand) / N)
        edges = list(g2.in_edges(nodes_rand[i]))+list(g2.out_edges(nodes_rand[i]))
        g2.remove_edges_from(edges)

        strong_adapt_degree = max(nx.strongly_connected_components(g3), key=len)
        strong_list_adapt_degree.append(len(strong_adapt_degree) / N)
        edges = list(g3.in_edges(adapt_degree[i])) + list(g3.out_edges(adapt_degree[i]))
        g3.remove_edges_from(edges)

        strong_degree = max(nx.strongly_connected_components(g4),key=len)
        strong_list_degree.append(len(strong_degree) / N)
        edges=list(g4.in_edges(degree[i]))+list(g4.out_edges(degree[i]))
        g4.remove_edges_from(edges)

        strong_degree_in = max(nx.strongly_connected_components(g1),key=len)
        strong_list_degree_in.append(len(strong_degree_in) / N)
        edges=list(g1.in_edges(degree_in[i]))+list(g1.out_edges(degree_in[i]))
        g1.remove_edges_from(edges)


        strong_degree_out = max(nx.strongly_connected_components(g8),key=len)
        strong_list_degree_out.append(len(strong_degree_out) / N)
        edges=list(g8.in_edges(degree_out[i]))+list(g8.out_edges(degree_out[i]))
        g8.remove_edges_from(edges)

        strong_MS = max(nx.strongly_connected_components(g5), key=len)
        strong_list_MS.append(len(strong_MS) / N)
        edges = list(g5.in_edges(minisum_nodes[i])) + list(g5.out_edges(minisum_nodes[i]))
        g5.remove_edges_from(edges)

        strong_pagerank = max(nx.strongly_connected_components(g6), key=len)
        strong_list_pagerank.append(len(strong_pagerank) / N)
        edges = list(g6.in_edges(pagerank_nodes[i])) + list(g6.out_edges(pagerank_nodes[i]))
        g6.remove_edges_from(edges)


        strong_DND = max(nx.strongly_connected_components(g7), key=len)
        strong_list_DND.append(len(strong_DND) / N)
        edges = list(g7.in_edges(str(DND_nodes[i]))) + list(g7.out_edges(str(DND_nodes[i])))
        g7.remove_edges_from(edges)



        strong_Core= max(nx.strongly_connected_components(g9), key=len)
        strong_list_Core.append(len(strong_Core) / N)
        edges = list(g9.in_edges(CoreHD_nodes[i])) + list(g9.out_edges(CoreHD_nodes[i]))
        g9.remove_edges_from(edges)

        if max(strong_list_degree_in[-1], strong_list_degree_out[-1]) <= 1 / N:
            break


    # print(strong_list_FINDER)
    val = 1 / N

    strong_list_rand += [val] * (N - i - 1)
    strong_list_degree += [val] * (N - i - 1)
    strong_list_adapt_degree += [val] * (N - i - 1)
    strong_list_MS += [val] * (N - i - 1)
    strong_list_pagerank += [val] * (N - i - 1)
    strong_list_DND += [val] * (N - i - 1)
    strong_list_Core += [val] * (N - i - 1)
    strong_list_degree_in += [val] * (N - i - 1)
    strong_list_degree_out += [val] * (N - i - 1)


    np.save('final_DN_result/'+filename+'_rand.npy',strong_list_rand)
    np.save('final_DN_result/'+filename+'_adpDegree.npy',strong_list_adapt_degree)
    np.save('final_DN_result/'+filename+'_degree.npy',strong_list_degree)
    np.save('final_DN_result/'+filename+'_MS.npy',strong_list_MS)
    np.save('final_DN_result/' + filename + '_PR.npy', strong_list_pagerank)
    np.save('final_DN_result/' + filename + '_DND.npy', strong_list_DND)
    np.save('final_DN_result/' + filename + '_Core.npy', strong_list_Core)
    np.save('final_DN_result/'+filename+'_ID.npy',strong_list_degree_in)
    np.save('final_DN_result/'+filename+'_OD.npy',strong_list_degree_out)

    epoch += 1
    pass
