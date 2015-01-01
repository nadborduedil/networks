import networkx as nx
import numpy as np
import time
from munkres import Munkres

class Matching:
    """
    this class is basically a bidirectional map
    the only reason I use it instead of a list of pairs or an ordinary map
    is that I want it to throw exception when i screw up and try to add the
    same thing twice
    """
    def __init__(self, items=None):
        self.ab = {}
        self.ba = {}
        if items:
            for a, b in items:
                self.add(a, b)

    def add(self, a, b):
        assert a not in self.ab and b not in self.ba
        self.ab[a] = b
        self.ba[b] = a

    def pop_a(self, a):
        b = self.ab.pop(a)
        self.ba.pop(b)
        return b

    def pop_b(self, b):
        a = self.ba.pop(b)
        self.ab.pop(a)

    def get_b(self, a):
        return self.ab.get(a)

    def get_a(self, b):
        return self.ba.get(b)

    def pop_a(self, b):
        x = self.ba.pop(b)
        self.ab.pop(x)
        return x

    def pop_b(self, a):
        x =  self.ab.pop(a)
        self.ba.pop(x)
        return x

    def items(self):
        return self.ab.items()

    def map_a(self, fun):
        new_items = [(fun(a), b) for a, b in self.items()]
        return Matching(new_items)

    def mab_b(self, fun):
        new_items = [(a, fun(b)) for a, b in self.items()]
        return Matching(new_items)

    def map_ab(self, fun_a, fun_b):
        new_items = [(fun_a(a), fun_b(b)) for a, b in self.items()]
        return Matching(new_items)

    def contains_a(self, a):
        return a in self.ab

    def contains_b(self, b):
        return b in self.ba

    def __iter__(self):
        return iter(self.ab.items())

    def __len__(self):
        return len(self.ab)

    def __str__(self):
        return str(self.ab.items())

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

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

def features_dict(graph, anchors):
    feats = {}
    for node in graph.nodes():
        dists = [nx.shortest_path_length(graph, node, anchor) for anchor in anchors]
        feats[node] = np.array(dists)
    return feats

def anchored_pagerank(graph, anchor):
    weights = dict((i, 0) for i in graph.nodes())
    weights[anchor] = 1
    return nx.pagerank_numpy(graph, personalization=weights)

def features_dict(graph, anchors, use_dist=True, use_pgrs=True,
                    use_pgr=True, use_comm=False, use_comm_centr=False):
    node_feats = {}
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


        node_feats[node] = np.array(feats)
    return node_feats


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

def dist(feature_vector_1, feature_vector_2):
    return ((feature_vector_1-feature_vector_2)**2).sum()

def make_cost_matrix(g1, g2, anchors1, anchors2, **kwargs):
    f1 = features_matrix(g1, anchors1, **kwargs)
    f2 = features_matrix(g2, anchors2, **kwargs)
    n1, n2 = len(f1), len(f2)
    return [[dist(f1[a], f2[b]) for b in range(n2)] for a in range(n1)]

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


