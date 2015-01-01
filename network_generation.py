import random
import numpy as np
import networkx as nx

import graph_matching as gm

def decimated_graph(g, p, q):
    """returns new graph with random nodes and edges removed

    First removes nodes with all their edges, then removes more edges
    :param g: original graph
    :param p: fraction of original nodes to retain
    :param q: fraction of edges to retain
    :return: decimated graph
    """
    nodes = g.nodes()
    n = len(nodes)
    n_nodes = set(random.sample(nodes, int(p*n)))
    n_edges = [(a, b) for a, b in g.edges() if a in n_nodes and b in n_nodes]

    n_edges = random.sample(n_edges, int(len(n_edges) * q))
    return nx.Graph(n_edges)


def kronecker_graph(g, k, add_self_edges=True, strip_self_edges=True):
    """returns Kronecker graph - tensor product of k copies of graph g

    Takes a graph g and int k and multiplies g's adjacency matrix
    by itself k times, returns resulting graph. For some reason graphs gotten
    this way are nicest when the initial graph has as self edge on every node,
    so by default this function adds self edges before the multiplication and
    strips them afterwards.
    :param g: initial graph
    :param k: exponent
    :param strip_self_edges:
    :param add_self_edges:
    :return: kronecker graph g ** k
    """

    adj = nx.adjacency_matrix(g).todense()
    if add_self_edges:
        for i in range(len(adj)):
            adj[i, i] = 1
    mat = adj
    for i in range(k-1):
        mat = np.kron(mat, adj)
    if strip_self_edges:
        for i in range(len(mat)):
            mat[i, i] = 0
    return nx.Graph(mat)

def permute_graphs(a, b):
    """takes two graphs that have some nodes in commmon and

    This function takes a pair of graphs a, b with some corresponding nodes
    - having the same label. Before running any matching algorithms them, their
    nodes and edges need to be shuffled because otherwise some algorithms
    may (inadvertently) match graphs by node label and give false impression
    of accuracy. This function permutes nodes and edges in on of the graphs
    and returns both graphs + the true matching.
    :param a: graph
    :param b: the other graph
    :return: graph a' (which is =a in this implementation), graph b', matching
    """
    nodes = b.nodes()
    permuted_nodes = np.random.permutation(nodes)

    # matching of all labels of nodes in graph b to their new values
    match = gm.Matching(zip(nodes, permuted_nodes))
    new_edges = [(match.get_b(x), match.get_b(y)) for x, y in b.edges()]
    permuted_edges = [(x,y) for x, y in np.random.permutation(new_edges)]
    unneeded_nodes = set(nodes).difference(set(a.nodes()))
    for node in unneeded_nodes:
        match.pop_b(node)
    return a, nx.Graph(permuted_edges), match

def get_anchors_candidates(a, b, match, n_anchors, n_candidates):
    """returns randomly selected anchors and candidate sets

    Randomly selects a set of pairs of corresponding nodes - anchors. For every
    node that in a that has a counterpart in b (and is not already an anchor)
    randomly selects a list of candidates, the list always contains true match.
    Candidate sets need not be disjoint.
    :param a: facebook
    :param b: linkedin
    :param match: true match between nodes of the graphs (may be undefined for
        some nodes)
    :param n_anchors: number of anchors - nodes with a match that is known
        upfront
    :param n_candidates: number of candidates per node
    :return: anchors, candidates
    """
    a_nodes = set(a.nodes())
    b_nodes = set(b.nodes())
    anchors = random.sample(match.items(), n_anchors)
    a_anchors = set([x for x, _ in anchors])
    b_anchors = set([y for _, y in anchors])
    a_nodes.difference_update(a_anchors)
    b_nodes.difference_update(b_anchors)
    candidates = {}
    for node in a_nodes:
        if match.contains_a(node):
            cands = random.sample(b_nodes, n_candidates-1)
            cands.append(match.get_b(node))
            candidates[node] = list(set(cands))
    return anchors, candidates
