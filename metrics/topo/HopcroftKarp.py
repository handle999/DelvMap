import networkx as nx


def HopcroftKarpCount(bigraph):
    G = nx.Graph()

    # 构造二分图节点集合
    holes_set = set(bigraph.keys())
    marble_set = set()
    for hole, marble_ids in bigraph.items():
        marble_set.update(marble_ids)

    # 添加所有节点（并注明二分图左右）
    G.add_nodes_from(holes_set, bipartite=0)
    G.add_nodes_from(marble_set, bipartite=1)

    # 添加所有边
    for hole, marble_ids in bigraph.items():
        for marble_id in marble_ids:
            G.add_edge(hole, marble_id)

    # 求最大匹配
    matches = nx.bipartite.maximum_matching(G, top_nodes=holes_set)
    # 注意：networkx 的匹配结果是双向的，因此只对一边计数
    matchedNum = sum(1 for u in matches if u in holes_set)

    # 返回匹配的数量
    return matchedNum
