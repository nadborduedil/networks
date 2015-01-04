import numpy as np
import networkx as nx
from munkres import Munkres
from graph_matching import Matching, features_dict, dist

def match_hungarian_x(cost_matrix):
    return Matching(Munkres().compute(cost_matrix))

def match_heuristic_x(cost_matrix):
    matches = Matching()
    for a_node, costs in enumerate(cost_matrix):
        candidates = [(b_node, cost) for b_node, cost in enumerate(costs) if
                      not matches.contains_b(b_node)]
        b_match = min(candidates, key=lambda x: x[1])[0]
        matches.add(a_node, b_match)
    return matches


def match_graphs_x(g1, g2, anchors1, anchors2, hungarian=False):
    f1 = features_dict(g1, anchors1)
    f2 = features_dict(g2, anchors2)

    labels_1 = f1.keys()
    labels_2 = f2.keys()
    vectors_1 = f1.values()
    vectors_2 = f2.values()
    lmap_1 = dict(list(enumerate(labels_1)))
    lmap_2 = dict(list(enumerate(labels_2)))

    cost_matrix = [[dist(v1, v2) for v2 in vectors_2] for v1 in vectors_1]
    if hungarian:
        matches = match_hungarian_x(cost_matrix)
    else:
        matches = match_heuristic_x(cost_matrix)
    return matches.map_ab(lambda x: lmap_1[x], lambda y: lmap_2[y]), cost_matrix

def make_cost_matrix(g1, g2, anchors1, anchors2, **kwargs):
    f1 = features_matrix(g1, anchors1, **kwargs)
    f2 = features_matrix(g2, anchors2, **kwargs)
    n1, n2 = len(f1), len(f2)
    return [[dist(f1[a], f2[b]) for b in range(n2)] for a in range(n1)]



def permuted_graph(g):
    nodes = g.nodes()
    permuted_nodes = np.random.permutation(nodes)
    p = permutation = Matching(list(zip(nodes, permuted_nodes)))
    edges = [(p.get_b(x), p.get_b(y)) for x, y in
             np.random.permutation(g.edges())]

    return nx.Graph(data=edges), permutation

def score(matching, feat_dict_1, feat_dict_2):
    return sum(dist(feat_dict_1[a], feat_dict_2[b]) for a, b in matching)


def score_x(matching, cost_matrix):
    return sum(cost_matrix[x][y] for x, y in matching.items())

def hits_misses(matching, true_matching):
    hits, misses = 0, 0
    for a, b in matching:
        if true_matching.get_b(a) == b:
            hits += 1
        else:
            misses += 1
    return hits, misses


def features_matrix(graph, anchors, use_dist=True, use_pgrs=True,
                    use_pgr=True, use_comm=False, use_comm_centr=False):
    node_feats = []
    n = len(graph)
    if use_dist:
        dists = nx.all_pairs_shortest_path_length(graph)
    if use_pgr:
        pageranks = nx.pagerank_numpy(graph)
    if use_pgrs:
        pgr_anchor = [anchored_pagerank(graph, anchor) for anchor in anchors]
    if use_comm_centr:
        communicability_centrality = nx.communicability_centrality(graph)
    if use_comm:
        communicability = nx.communicability(graph)

    for node in graph.nodes():
        assert node == len(node_feats)
        feats = []
        if use_dist:
            feats += [dists[node][anchor] for anchor in anchors]
        if use_pgrs:
            feats += [pgr[node]*n for pgr in pgr_anchor]
        if use_pgr:
            feats.append(pageranks[node]*n)
        if use_comm_centr:
            feats.append(communicability_centrality[node])
        if use_comm:
            feats += [communicability[node][anchor] for anchor in anchors]


        node_feats.append(np.array(feats))
    return node_feats