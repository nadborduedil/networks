import graph_matching as gm


class BasicMatcher:
    def __init__(self, use_dist=False, use_pgrs=True, use_pgr=True,
                 use_comm=False,
                 use_comm_centr=False):
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

        threshold = 1
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


    def generate_matches1(self, a, b, anchors, candidate_dict, true_match):
        a_anchors = [x for x, _ in anchors]
        b_anchors = [y for _, y in anchors]
        a_feats = gm.features_dict(a, a_anchors)
        b_feats = gm.features_dict(b, b_anchors)

        threshold = 1
        matches = []
        for position, candidates in candidate_dict.items():
            p_feats = a_feats[position]
            scored_cands = [(c, gm.dist(p_feats, b_feats[c])) for c in
                            candidates]
            scored_cands = sorted(scored_cands, key=lambda x: x[1])
            c, score = min(scored_cands, key=lambda x: x[1])
            true = true_match.get_b(position)
            # print position, c, true, "%.2f" % score, " :)" if true==c else ""
            # print scored_cands
            # print candidates
            # if score < threshold:
            matches.append((position, c, score, true))
        return matches