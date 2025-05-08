import numpy as np
import networkx as nx
import os

def findfile(directory, file_prefix):
    filenames = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames

#dir = "bar_source/"
#file_pre = "SF_1000_"

#dir = "SF_new_lamda2-3.5/"          #SF网络数据
#dir = "SF_ER_FINDER/SF/"          #SF网络数据
#dir = "ER_new_d_3-20/"          #ER网络数据
#dir = "SF_ER_FINDER/ER/"          #ER网络数据
dir = "real_network/"           #真实网络数据

file_pre = ""

filenames = findfile(dir, file_pre)

epoch = 0
for file in filenames:
    print('epoch:', epoch)
    print(file)
    filename = file  # [:-4]
    g = nx.read_graphml(dir + filename)  # 读取graphhml形式储存的图
    g.remove_edges_from(nx.selfloop_edges(g))  # 去掉指向自己的自环边

    nodes = list(g.nodes)
    N = len(nodes)
    print(N)
    edges = g.edges()
    a, b = zip(*edges)
    A = np.array(a)
    B = np.array(b)
    print(A)
    print(B)
    back_nodes = []
    selected = []

    # FINDER方法取节点
    print("FINDER_directed")
    idx1 = np.load('FINDER_selected_directed/' + file + '.npy').astype(int)
    #print(idx1)
    nodes = [str(item) for item in idx1]
    #print(nodes)
    idx1 = nodes + list(set(g.nodes()) - set(nodes))
    #print(idx1)

    print("FINDER_undirected")
    idx2 = np.load('FINDER_selected_undirected/' + file + '.npy').astype(int)
    #print(idx2)
    nodes = [str(item) for item in idx2]
    #print(nodes)
    idx2 = nodes + list(set(g.nodes()) - set(nodes))
    #print(idx2)


    g1 = g.copy()
    g2 = g.copy()

    strong_list_FINDER = []
    strong_list_FINDER_un = []


    for i in range(min(len(idx1), len(idx2))):

        strong_FINDER = max(nx.strongly_connected_components(g1), key=len)
        strong_list_FINDER.append(len(strong_FINDER) / N)
        edges = list(g1.in_edges(str(idx1[i]))) + list(g1.out_edges(str(idx1[i])))
        g1.remove_edges_from(edges)

        strong_FINDER = max(nx.strongly_connected_components(g2), key=len)
        strong_list_FINDER_un.append(len(strong_FINDER) / N)
        edges = list(g2.in_edges(str(idx2[i]))) + list(g2.out_edges(str(idx2[i])))
        g2.remove_edges_from(edges)

    val = 1 / N

    strong_list_FINDER += [val] * (N - i - 1)       #strong_list_FINDER列表添加(N - i - 1) 个1 / N
    strong_list_FINDER_un += [val] * (N - i - 1)

    np.save('FINDER_directed_result/' + filename + '_FD.npy', strong_list_FINDER)
    np.save('FINDER_directed_un_result/' + filename + '_FD_un.npy', strong_list_FINDER_un)

    epoch += 1