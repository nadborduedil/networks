import networkx as nx
import numpy as np
from utils import memo, CACHE_PATH
from persistent_dict import PersistentDict
"""TODO: feature extraction should return numpy arrays not dicts this will
simplify all operations A LOT. Reuquires changing network_generation to
return graphs with nodes labeled by numbers in range(n)"""

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

def normalized_dict(d, order=2):
    n = len(d)
    if n == 0:
        return {}
    norm = sum(v**order for v in d.values())**(1./order)
    if norm == 0:
        return dict((k, 1./n) for k in d.keys())
    else:
        return dict((k, v/norm) for k, v in d.items())

def normalized(a, order=2, axis=-1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def _dists_to_anchors(graph, anchors, normalize=False):
    dists = {}
    for node in graph.nodes():
        d = []

        for anchor in anchors:
            d.append(nx.shortest_path_length(graph, node, anchor))
        if normalize:
            d = normalized(d)
        dists[node] = np.array(d)
    return dists

class DistExtractor:
    def __init__(self, normalize=False, weight=None):
        self.normalize = normalize
        self.weight = weight

    def extract(self, graph, anchors):
        return _dists_to_anchors(graph, anchors, self.normalize)


@memo({})
def _anchored_pagerank(graph, anchors, normalize=False):
    n = len(graph)
    pgranks = dict((node, {}) for node in graph.nodes())
    for anchor in anchors:
        weights = dict((i, 0) for i in graph.nodes())
        weights[anchor] = 1
        pgr = nx.pagerank_numpy(graph, personalization=weights)
        for node, v in pgr.items():
            pgranks[node][anchor] = v

    for node, pgr in pgranks.items():
        pgranks[node] = d = np.array([pgr[a] for a in anchors])
        if normalize:
            pgranks[node] = normalized(d)
            # d[n] = normalized_dict(pgr)
    return pgranks

class AnchoredPageranksExtractor:
    def __init__(self, normalize=False, weight=None):
        self.normalize = normalize
        self.weight = weight

    def extract(self, graph, anchors):
        return _anchored_pagerank(graph, anchors, self.normalize)


# this sucks. either there's a fundamental bug or communicability is utterly
# unsuited for this. Also - im using communicability_exp, because the other
# one uses spectral decomposition and therefore takes cubic time which is
# unbearably slow, other than that the both suck the same amount of balls.
# im only keeping it in as a reminder not to try it again.
# <suckage>
@memo({})
def _communicability_to_anchors(graph, anchors, normalize=False):
    comm = {}
    for node, comm_dict in nx.communicability_exp(graph).items():
        d = np.array([comm_dict[a] for a in anchors])
        if normalize:
            d = normalized(d)
        comm[node] = d
    return comm

class CommunicabilityExtractor:
    def __init__(self, normalize=False, weight=None):
        self.normalize = normalize
        self.weight = weight

    def extract(self, graph, anchors):
        return _communicability_to_anchors(graph, anchors, self.normalize)
# </suckage>

# WOW this is awesome - runs really fast and gives great results
def _authority_matrix(graph, anchors, normalize):
    aut = {}
    node_to_num = dict((node, i) for i, node in enumerate(graph.nodes()))
    num_to_node = dict(enumerate(graph.nodes()))
    aut_mat = nx.authority_matrix(graph)
    for num, node in enumerate(graph.nodes()):
        d = []
        for anchor in anchors:
            a_num = node_to_num[anchor]
            d.append(aut_mat[num, a_num])
        if normalize:
            d = normalized(d)
        aut[node] = np.array(d)
    return aut

class AuthorityExtractor:
    def __init__(self, normalize=False, weight=None):
        self.normalize = normalize
        self.weight = weight

    def extract(self, graph, anchors):
        return _authority_matrix(graph, anchors, self.normalize)


def _google_matrix(graph, anchors, normalize):
    aut = {}
    node_to_num = dict((node, i) for i, node in enumerate(graph.nodes()))
    num_to_node = dict(enumerate(graph.nodes()))
    aut_mat = nx.google_matrix(graph)
    for num, node in enumerate(graph.nodes()):
        d = []
        for anchor in anchors:
            a_num = node_to_num[anchor]
            d.append(aut_mat[num, a_num])
        if normalize:
            d = normalized(d)
        aut[node] = np.array(d)
    return aut

class GoogleExtractor:
    def __init__(self, normalize=False, weight=None):
        self.normalize = normalize
        self.weight = weight

    def extract(self, graph, anchors):
        return _google_matrix(graph, anchors, self.normalize)



def anchored_pagerank(graph, anchor):
    weights = dict((i, 0) for i in graph.nodes())
    weights[anchor] = 1
    return nx.pagerank_numpy(graph, personalization=weights)

@memo(PersistentDict(CACHE_PATH+"/graph_matching.pageranks_to_anchors.db"))
# @memo({})
def pageranks_to_anchors(graph, anchors):
    return [anchored_pagerank(graph, anchor) for anchor in anchors]

@memo(PersistentDict(CACHE_PATH+"/graph_matching.dists_to_anchors.db"))
# @memo({})
def dists_to_anchors(graph, anchors):
    dists = {}
    for node in graph.nodes():
        d = {}
        dists[node] = d
        for anchor in anchors:
            d[anchor] = nx.shortest_path_length(graph, node, anchor)

    return dists

def features_dict(graph, anchors, use_dist=True, use_pgrs=True,
                    use_pgr=True, use_comm=False, use_comm_centr=False):
    node_feats = {}
    n = len(graph)
    if use_dist:
        # dists = nx.all_pairs_shortest_path_length(graph)
        dists = dists_to_anchors(graph, anchors)
    if use_pgr:
        pageranks = nx.pagerank_numpy(graph)
    if use_pgrs:
        # pgr_anchor = [anchored_pagerank(graph, anchor) for anchor in anchors]
        pgr_anchor = pageranks_to_anchors(graph, anchors)
    if use_comm_centr:
        communicability_centrality = nx.communicability_centrality(graph)
    if use_comm:
        communicability = nx.communicability(graph)

    for node in graph.nodes():
        feats = []
        if use_dist:
            feats += [dists[node][anchor] for anchor in anchors]
        if use_pgrs:
            feats += [pgr_anchor[anchor][node]*n
                      for anchor in range(len(anchors))]
            # feats += [pgr[node]*n for pgr in pgr_anchor]
        if use_pgr:
            feats.append(pageranks[node]*n)
        if use_comm_centr:
            feats.append(communicability_centrality[node])
        if use_comm:
            feats += [communicability[node][anchor] for anchor in anchors]


        node_feats[node] = np.array(feats)
    return node_feats



def dist(feature_vector_1, feature_vector_2):
    return ((feature_vector_1-feature_vector_2)**2).sum()


