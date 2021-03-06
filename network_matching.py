import numpy as np
import graph_matching as gm

class BasicMatcher:
    def __init__(self, use_dist=False, use_pgrs=True, use_pgr=True,
                 use_comm=False, use_comm_centr=False):
        self.use_dist = use_dist
        self.use_pgrs = use_pgrs
        self.use_pgr = use_pgr
        self.use_comm = use_comm
        self.use_comm_centr = use_comm_centr


    def generate_matches(self, a, b, anchors, candidate_dict):
        a_anchors = tuple([x for x, _ in anchors])
        b_anchors = tuple([y for _, y in anchors])
        a_feats = gm.features_dict(a, a_anchors, self.use_dist, self.use_pgrs,
                                   self.use_pgr, self.use_comm,
                                   self.use_comm_centr)
        b_feats = gm.features_dict(b, b_anchors, self.use_dist, self.use_pgrs,
                                   self.use_pgr, self.use_comm,
                                   self.use_comm_centr)

        matches = []
        for position, candidates in candidate_dict.items():
            p_feats = a_feats[position]
            scored_cands = [(c, gm.dist(p_feats, b_feats[c])) for c in
                            candidates]
            scored_cands = sorted(scored_cands, key=lambda x: x[1])
            c, score = min(scored_cands, key=lambda x: x[1])
            # print position, c, true, "%.2f" % score, " :)" if true==c else ""
            # print scored_cands
            # print candidates
            # if score < threshold:
            matches.append((position, c, score))
        return matches


class BasicMatcher2:
    def __init__(self, feature_extractors, threshold=1):
        self.extractors = feature_extractors
        self.threshold = threshold


    def dist(self, a_feats, b_feats):
        return sum((extractor.weight or 1) * gm.dist(a, b) for a, b, extractor
                   in zip(a_feats, b_feats, self.extractors))

    def generate_matches(self, a, b, anchors, candidate_dict):
        a_anchors = tuple([x for x, _ in anchors])
        b_anchors = tuple([y for _, y in anchors])
        a_feats_per_extr = [e.extract(a, a_anchors) for e in self.extractors]
        a_feats_per_node = dict(
            (node, [feats[node] for feats in a_feats_per_extr]) for node in
            a.nodes())
        b_feats_per_extr = [e.extract(b, b_anchors) for e in self.extractors]
        b_feats_per_node = dict(
            (node, [feats[node] for feats in b_feats_per_extr]) for node in
            b.nodes())

        anchor_dists = [
            self.dist(a_feats_per_node[a_anc], b_feats_per_node[b_anc]) for
            a_anc, b_anc in zip(a_anchors, b_anchors)]
        threshold = np.percentile(anchor_dists, self.threshold * 100)
        matches = []
        for position, candidates in candidate_dict.items():
            p_feats = a_feats_per_node[position]
            scored_cands = [(c, self.dist(p_feats, b_feats_per_node[c])) for
                            c in candidates]
            scored_cands = sorted(scored_cands, key=lambda x: x[1])
            c, score = min(scored_cands, key=lambda x: x[1])
            matches.append((position, c, score))
        return matches