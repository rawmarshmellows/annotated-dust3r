def make_pairs(imgs, scene_graph="complete", symmetrize=True):
    pairs = []
    if scene_graph == "complete":  # complete graph
        for i in range(len(imgs)):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))

    if symmetrize:
        pairs += [(img2, img1) for img1, img2 in pairs]
    return pairs
