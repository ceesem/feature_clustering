import pandas as pd
import numpy as np
from sklearn import decomposition
from scipy import stats


def percentile_above_zero(X, percentile=50):
    out = []
    for x in X:
        out.append(np.percentile(x[x > 0], percentile))
    return np.array(out)


def assemble_features_from_data(df, n_syn_comp=5, n_branch_comp=3):
    df = df.copy()

    df["tip_len_dist_dendrite_p50"] = df["tip_len_dist_dendrite"].apply(
        lambda x: np.percentile(x, 75)
    )
    df["tip_tort_dendrite_p50"] = df["tip_tort_dendrite"].apply(
        lambda x: np.percentile(x, 75)
    )

    df["syn_size_distribution_soma_p50"] = df["syn_size_distribution_soma"].apply(
        np.median
    )
    df["syn_dist_distribution_dendrite_p50"] = df[
        "syn_dist_distribution_dendrite"
    ].apply(np.median)
    df["syn_size_distribution_dendrite_p50"] = df[
        "syn_size_distribution_dendrite"
    ].apply(np.median)
    df["syn_size_distribution_dendrite_p10"] = df[
        "syn_size_distribution_dendrite"
    ].apply(lambda x: np.percentile(x, 10))
    df["syn_size_distribution_dendrite_p90"] = df[
        "syn_size_distribution_dendrite"
    ].apply(lambda x: np.percentile(x, 90))
    df["syn_size_distribution_dendrite_dyn_range"] = (
        df["syn_size_distribution_dendrite_p90"]
        - df["syn_size_distribution_dendrite_p10"]
    )
    df["syn_size_dendrite_cv"] = df["syn_size_distribution_dendrite"].apply(
        np.std
    ) / df["syn_size_distribution_dendrite"].apply(np.mean)

    df["syn_dist_distribution_basal_p50"] = df["syn_dist_distribution_basal"].apply(
        np.median
    )
    df["syn_size_distribution_soma_p50"] = df["syn_size_distribution_soma"].apply(
        np.median
    )
    df["syn_size_distribution_basal_p50"] = df["syn_size_distribution_basal"].apply(
        np.median
    )
    df["syn_size_distribution_basal_p10"] = df["syn_size_distribution_basal"].apply(
        lambda x: np.percentile(x, 10)
    )
    df["syn_size_distribution_basal_p90"] = df["syn_size_distribution_basal"].apply(
        lambda x: np.percentile(x, 90)
    )
    df["syn_size_distribution_basal_dyn_range"] = (
        df["syn_size_distribution_dendrite_p90"]
        - df["syn_size_distribution_dendrite_p10"]
    )

    df["syn_depth_dist_p5"] = df["syn_depth_dist_all"].apply(
        lambda x: np.percentile(x, 5)
    )
    df["syn_depth_dist_p95"] = df["syn_depth_dist_all"].apply(
        lambda x: np.percentile(x, 95)
    )
    df["syn_depth_extent"] = df["syn_depth_dist_p95"] - df["syn_depth_dist_p5"]

    svd_br = decomposition.TruncatedSVD(n_branch_comp)
    dbr = np.vstack(df["branches_dist"].values)
    Xbr = svd_br.fit_transform(dbr)
    for ii in range(Xbr.shape[1]):
        df[f"branch_svd{ii}"] = Xbr[:, ii]

    syn_pca = decomposition.SparsePCA(n_components=n_syn_comp)
    pl_dat = np.vstack(df["syn_count_depth_dendrite"].values)
    keep_dat_cols = np.sum(pl_dat, axis=0) > 0
    pl_dat_z = stats.zscore(pl_dat[:, keep_dat_cols])
    X = syn_pca.fit_transform(pl_dat_z)
    for ii in range(X.shape[1]):
        df[f"syn_count_pca{ii}"] = X[:, ii]

    pl_depth = np.vstack(df["path_length_depth_dendrite"].values)
    sc_depth = np.vstack(df["syn_count_depth_dendrite"].values)
    keep_cols = pl_depth.sum(axis=0) > 0
    density_nan = sc_depth[:, keep_cols] / pl_depth[:, keep_cols]
    density_nan[np.isnan(density_nan)] = 0
    density_nan[np.isinf(density_nan)] = 0
    df["max_density"] = percentile_above_zero(density_nan, 75)

    dat_cols = [
        "tip_len_dist_dendrite_p50",
        "tip_tort_dendrite_p50",
        "num_syn_dendrite",
        "num_syn_soma",
        "path_length_dendrite",
        "radial_extent_dendrite",
        "syn_dist_distribution_dendrite_p50",
        "syn_size_distribution_soma_p50",
        "syn_size_distribution_dendrite_p50",
        "syn_size_distribution_dendrite_dyn_range",
        # "syn_size_dendrite_cv",
        "syn_depth_dist_p5",
        # "syn_depth_dist_p95",
        "syn_depth_extent",
        "max_density",
    ]

    for ii in range(X.shape[1]):
        dat_cols.append(f"syn_count_pca{ii}")
    for ii in range(Xbr.shape[1]):
        dat_cols.append(f"branch_svd{ii}")

    return_cols = ["root_id", "soma_depth"] + dat_cols
    return df[return_cols], dat_cols, syn_pca, svd_br, keep_dat_cols