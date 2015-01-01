import graph_matching as gm


def generate_matches(a, b, anchors, candidate_dict, true_match):
    a_anchors = [x for x, _ in anchors]
    b_anchors = [y for _, y in anchors]
    a_feats = gm.features_dict(a, a_anchors)
    b_feats = gm.features_dict(b, b_anchors)

    threshold = 1
    matches = []
    for position, candidates in candidate_dict.items():
        p_feats = a_feats[position]
        scored_cands = [(c, gm.dist(p_feats, b_feats[c])) for c in candidates]
        scored_cands = sorted(scored_cands, key=lambda x: x[1])
        c, score = min(scored_cands, key=lambda x: x[1])
        true = true_match.get_b(position)
        # print position, c, true, "%.2f" % score, " :)" if true==c else ""
        # print scored_cands
        # print candidates
        # if score < threshold:
        matches.append((position, c, score, true))
    return matches