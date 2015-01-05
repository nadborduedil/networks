from __future__ import division

import numpy as np

import networkx as nx
import network_generation as netgen
from utils import memo, CACHE_PATH
from persistent_dict import PersistentDict


class TestCase:
    def __init__(self, g_a, g_b, anchors, cands, true_match, name):
        self.g_a = g_a
        self.g_b = g_b
        self.anchors = anchors
        self.cands = cands
        self.true_match = true_match
        self.name = name

@memo(PersistentDict(CACHE_PATH+"/testing.kronecker_regular.db"))
# @memo({})
def kronecker_regular(seed_degree=3, seed_nodes=8, exponent=2,
                      a_nodes_left=0.8, a_edges_left=0.8, b_nodes_left=0.8,
                      b_edges_left=0.8, anchors=5, candidates=8,
                      random_seed=0):
    other_seed = random_seed + 1
    seed_graph = nx.random_regular_graph(seed_degree, seed_nodes, random_seed)
    g = netgen.kronecker_graph(seed_graph, exponent)
    g_a = netgen.decimated_graph(g, a_nodes_left, a_edges_left, random_seed)
    g_b = netgen.decimated_graph(g, b_nodes_left, b_edges_left, other_seed)
    g_a, g_b, true_match = netgen.permute_graphs(g_a, g_b, random_seed)
    anchors, cands = netgen.get_anchors_candidates(g_a, g_b, true_match,
                                                   anchors, candidates,
                                                   random_seed)
    anchors = tuple(anchors)
    name = "kronecker_regular(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" % (
        seed_degree, seed_nodes, exponent, a_nodes_left, a_edges_left,
        b_nodes_left, b_edges_left, anchors, candidates, random_seed)
    return TestCase(g_a, g_b, anchors, cands, true_match, name)

@memo(PersistentDict(CACHE_PATH+"/testing.star_test_case.db"))
# @memo({})
def star_test_case(exponent=3,
                      a_nodes_left=0.8, a_edges_left=0.8, b_nodes_left=0.8,
                      b_edges_left=0.8, anchors=5, candidates=8,
                      random_seed=0):
    other_seed = random_seed + 1
    seed_graph = nx.Graph([(0,1), (0,2), (0,3), (0,4)], name="star")
    g = netgen.kronecker_graph(seed_graph, exponent)
    g_a = netgen.decimated_graph(g, a_nodes_left, a_edges_left, random_seed)
    g_b = netgen.decimated_graph(g, b_nodes_left, b_edges_left, other_seed)
    g_a, g_b, true_match = netgen.permute_graphs(g_a, g_b, random_seed)
    anchors, cands = netgen.get_anchors_candidates(g_a, g_b, true_match,
                                                   anchors, candidates,
                                                   random_seed)
    anchors = tuple(anchors)
    name = "star_case(%s, %s, %s, %s, %s, %s, %s, %s)" % (
        exponent, a_nodes_left, a_edges_left,
        b_nodes_left, b_edges_left, anchors, candidates, random_seed)
    return TestCase(g_a, g_b, anchors, cands, true_match, name)


def test(matcher, test_case):
    t = test_case
    matches = matcher.generate_matches(t.g_a, t.g_b, t.anchors, t.cands)

    total = len(t.cands)
    tru = 0
    fal = 0
    for match in matches:
        a_node = match[0]
        b_node = match[1]
        if b_node == t.true_match.get_b(a_node):
            tru += 1
        else:
            fal += 1

    return total, tru, fal


def validate_test_case(test_case):
    return nx.is_connected(test_case.g_a) and nx.is_connected(test_case.g_b)


def test_all(matcher, test_cases):
    results = np.array([test(matcher, tc) for tc in test_cases])
    totals = results[:, 0]
    trues = results[:, 1]

    falses = results[:, 2]
    true_rates = trues / totals
    false_rates = falses / totals

    return true_rates.mean(), false_rates.mean()


