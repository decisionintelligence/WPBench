import numpy as np

def _augment_align(part_dist, num_pad):
    scores = part_dist.mean(axis=0)
    idx = np.argsort(-scores)
    selected = []
    for i in idx:
        if scores[i] <= 0:
            break
        selected.append(i)
        if len(selected) >= num_pad:
            break
    if len(selected) < num_pad:
        extra = [i for i in range(part_dist.shape[1]) if i not in selected]
        for i in extra:
            selected.append(i)
            if len(selected) >= num_pad:
                break
    return np.array(selected[:num_pad], dtype=int)


def _reorder_data(parts_idx, adj, sps):
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if sps - part_idx.shape[0] > 0:
            local_part_idx = _augment_align(part_dist, sps - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx
        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + sps * i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])
    return ori_parts_idx, reo_parts_idx, reo_all_idx


def _kd_tree(locations, times, axis):
    sorted_idx = np.argsort(locations[axis])
    part1 = np.sort(sorted_idx[: locations.shape[1] // 2])
    part2 = np.sort(sorted_idx[locations.shape[1] // 2 :])
    if times == 1:
        return [part1, part2]
    parts = []
    left_parts = _kd_tree(locations[:, part1], times - 1, axis ^ 1)
    right_parts = _kd_tree(locations[:, part2], times - 1, axis ^ 1)
    for part in left_parts:
        parts.append(part1[part])
    for part in right_parts:
        parts.append(part2[part])
    return parts


def _load_spatial_managed_indices(locations, adj, recurtimes, sps):
    parts_idx = _kd_tree(locations, recurtimes, 0)
    return _reorder_data(parts_idx, adj, sps)