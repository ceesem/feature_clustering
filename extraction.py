import os
import orjson
import json
import pandas as pd
import numpy as np
from meshparty import meshwork
from .io_utils import anno_mask_dict, load_root_id
from scipy import sparse

radius_bins = np.arange(20, 300, 30)
radius_bin_width = 10


def make_depth_bins(height_bounds, spacing=50):
    return np.linspace(height_bounds[0], height_bounds[1], spacing)


def tip_len_dist(nrn, compartment=None):
    eps = nrn.end_points
    if compartment is not None:
        eps = eps[nrn.anno[anno_mask_dict[compartment]].mesh_mask[eps]]
    if len(eps) == 0:
        return None
    return nrn.distance_to_root(eps) / 1000


def tip_tort(nrn, compartment=None):
    eps = nrn.end_points
    if compartment is not None:
        eps = eps[nrn.anno[anno_mask_dict[compartment]].mesh_mask[eps]].to_skel_index
    if len(eps) == 0:
        return None
    dtr = nrn.skeleton.distance_to_root[eps] / 1000
    euc_dist = (
        np.linalg.norm(
            (
                nrn.skeleton.vertices[eps]
                - np.atleast_2d(nrn.skeleton.vertices[nrn.skeleton.root])
            ),
            axis=1,
        )
        / 1000
    )
    tort = dtr / euc_dist
    return tort


def num_syn(nrn, compartment=None):
    if compartment is None:
        df = nrn.anno.post_syn.df
    else:
        df = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).df
    return len(df)


def syn_size_distribution(nrn, compartment=None):
    if compartment is None:
        df = nrn.anno.post_syn.df
    else:
        df = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).df
    return df["size"].values


def syn_dist_distribution(nrn, compartment=None):
    if compartment is None:
        minds = nrn.anno.post_syn.mesh_index
    else:
        minds = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).mesh_index
    return nrn.distance_to_root(minds)


def syn_depth_dist(nrn, compartment=None):
    if compartment is None:
        df = nrn.anno.post_syn.df
    else:
        df = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).df
    if len(df) > 0:
        return np.array(df["ctr_pt_position"].apply(lambda x: 4 * x[1] / 1_000))
    else:
        return np.ascontiguousarray([])


def _is_between(xs, a, b):
    return np.logical_and(xs > a, xs <= b)


def _branches_between(nrn, d_a, d_b, min_thresh):
    gap = _is_between(nrn.skeleton.distance_to_root, d_a, d_b)
    G = nrn.skeleton.csgraph_binary_undirected
    _, ccs = sparse.csgraph.connected_components(G[:, gap][gap])
    _, nvs = np.unique(ccs, return_counts=True)
    return sum(nvs > min_thresh)


def branches_between(nrn, d_a, d_b, min_thresh=1, compartment=None):
    if compartment is None:
        return _branches_between(nrn, d_a, d_b, min_thresh)
    else:
        try:
            with nrn.mask_context(
                nrn.anno[anno_mask_dict[compartment]].mesh_mask
            ) as nmc:
                return _branches_between(nmc, d_a, d_b, min_thresh)
        except:
            return None


def branches_with_distance(nrn, radius_bins, radius_bin_width, compartment=None):
    return [
        branches_between(
            nrn, d_a * 1_000, 1_000 * (d_a + radius_bin_width), compartment=compartment
        )
        for d_a in radius_bins
    ]


def path_length(nrn, compartment=None):
    try:
        with nrn.mask_context(nrn.anno[anno_mask_dict[compartment]].mesh_mask) as nmc:
            return nmc.path_length() / 1_000
    except:
        return None


def horizontal_extent(nrn, compartment=None):
    center = nrn.skeleton.vertices[nrn.skeleton.root]
    try:
        with nrn.mask_context(nrn.anno[anno_mask_dict[compartment]].mesh_mask) as nmc:
            xz_vertices = nmc.skeleton.vertices[:, [0, 2]]
            if len(xz_vertices) == 0:
                return None

        dvec = xz_vertices - np.atleast_2d(center)[:, [0, 2]]
        d = np.linalg.norm(dvec, axis=1) / 1_000
        return np.percentile(d, 97)
    except:
        return None


def soma_depth(nrn):
    return nrn.skeleton.vertices[nrn.skeleton.root, 1] / 1_000


def make_depth_bins(height_bounds, nbins=50):
    return np.linspace(height_bounds[0], height_bounds[1], nbins)


def _node_weight(nrn):
    return (
        np.squeeze(np.array(np.sum(nrn.skeleton.csgraph_undirected, axis=0)) / 2)
        / 1_000
    )


def _path_length_binned(nrn, depth_bins):
    ws = _node_weight(nrn)
    sk_vert_y = nrn.skeleton.vertices[:, 1] / 1000
    lens = []
    for d_a, d_b in zip(depth_bins[:-1], depth_bins[1:]):
        lens.append(np.sum(ws[_is_between(sk_vert_y, d_a, d_b)]))
    return np.array(lens)


def path_length_binned(nrn, depth_bins, compartment=None):
    if compartment is None:
        return _path_length_binned(nrn, depth_bins)
    else:
        try:
            with nrn.mask_context(
                nrn.anno[anno_mask_dict[compartment]].mesh_mask
            ) as nmc:
                return _path_length_binned(nmc, depth_bins)
        except:
            return None
    pass


def _syn_count_binned(nrn, depth_bins):
    syn_y = (
        nrn.anno.post_syn.df["ctr_pt_position"].apply(lambda x: 4 * x[1] / 1000).values
    )
    n_syn = []
    for ymin, ymax in zip(depth_bins[:-1], depth_bins[1:]):
        n_syn.append(sum(_is_between(syn_y, ymin, ymax)))
    return np.array(n_syn)


def syn_count_binned(nrn, depth_bins, compartment=None):
    if compartment is None:
        return _syn_count_binned(nrn, depth_bins)
    else:
        try:
            with nrn.mask_context(
                nrn.anno[anno_mask_dict[compartment]].mesh_mask
            ) as nmc:
                return _syn_count_binned(nmc, depth_bins)
        except:
            return None
    pass


def extract_features_dict(nrn, radius_bins, radius_bin_width, depth_bins):
    return {
        "root_id": nrn.seg_id,
        "soma_depth": soma_depth(nrn),
        "tip_len_dist_dendrite": tip_len_dist(nrn, "dendrite"),
        "tip_len_dist_basal": tip_len_dist(nrn, "basal"),
        "tip_len_dist_apical": tip_len_dist(nrn, "apical"),
        "tip_tort_dendrite": tip_tort(nrn, "dendrite"),
        "tip_tort_basal": tip_tort(nrn, "basal"),
        "tip_tort_apical": tip_tort(nrn, "apical"),
        "num_syn_dendrite": num_syn(nrn, "dendrite"),
        "num_syn_basal": num_syn(nrn, "basal"),
        "num_syn_apical": num_syn(nrn, "apical"),
        "num_syn_soma": num_syn(nrn, "soma"),
        "syn_size_distribution_soma": syn_size_distribution(nrn, "soma"),
        "syn_size_distribution_dendrite": syn_size_distribution(nrn, "dendrite"),
        "syn_size_distribution_basal": syn_size_distribution(nrn, "basal"),
        "syn_size_distribution_apical": syn_size_distribution(nrn, "apical"),
        "syn_dist_distribution_dendrite": syn_dist_distribution(nrn, "dendrite"),
        "syn_dist_distribution_basal": syn_dist_distribution(nrn, "basal"),
        "syn_dist_distribution_apical": syn_dist_distribution(nrn, "apical"),
        "syn_depth_dist_all": syn_depth_dist(nrn, "dendrite"),
        "syn_depth_dist_basal": syn_depth_dist(nrn, "basal"),
        "syn_depth_dist_apical": syn_depth_dist(nrn, "apical"),
        "radial_extent_dendrite": horizontal_extent(nrn, "dendrite"),
        "radial_extent_basal": horizontal_extent(nrn, "basal"),
        "radial_extent_apical": horizontal_extent(nrn, "apical"),
        "path_length_dendrite": path_length(nrn, "dendrite"),
        "path_length_basal": path_length(nrn, "basal"),
        "path_length_apical": path_length(nrn, "apical"),
        "branches_dist": branches_with_distance(
            nrn, radius_bins, radius_bin_width, compartment="dendrite"
        ),
        "path_length_depth_dendrite": path_length_binned(
            nrn, depth_bins, compartment="dendrite"
        ),
        "path_length_depth_basal": path_length_binned(
            nrn, depth_bins, compartment="basal"
        ),
        "path_length_depth_apical": path_length_binned(
            nrn, depth_bins, compartment="apical"
        ),
        "syn_count_depth_dendrite": syn_count_binned(
            nrn, depth_bins, compartment="dendrite"
        ),
        "syn_count_depth_basal": syn_count_binned(nrn, depth_bins, compartment="basal"),
        "syn_count_depth_apical": syn_count_binned(
            nrn, depth_bins, compartment="apical"
        ),
        "success": True,
    }


def extract_features(nrn, feature_dir, project_paths, write=True, filename=None):
    try:
        height_bounds = np.load(f"{project_paths.data}/height_bounds_v1.npy")
        depth_bins = make_depth_bins(height_bounds)
        features = extract_features_dict(nrn, radius_bins, radius_bin_width, depth_bins)
    except:
        features = {
            "root_id": nrn.seg_id,
            "success": False,
        }

    if write:
        if filename is None:
            filename = f"{nrn.seg_id}"
        with open(f"{feature_dir}/{filename}.json", "wb") as f:
            f.write(orjson.dumps(features, option=orjson.OPT_SERIALIZE_NUMPY))

    return features


def extract_features_root_id(root_id, project_paths, feature_dir, rerun=False):
    if os.path.exists(f"{feature_dir}/{root_id}.json") and not rerun:
        with open(f"{feature_dir}/{root_id}.json") as f:
            dat = json.load(f)
            if dat["success"] is True:
                return True
    try:
        nrn = load_root_id(root_id, project_paths)
        features = extract_features(nrn, feature_dir, project_paths)
        return features["success"]
    except:
        return False


def extract_features_mp(root_ids, project_paths, feature_dir, nodes=8):
    from pathos.pools import ProcessPool

    pool = ProcessPool(nodes=nodes)
    return np.array(
        pool.map(
            extract_features_root_id,
            root_ids,
            [project_paths] * len(root_ids),
            [feature_dir] * len(root_ids),
        )
    )
