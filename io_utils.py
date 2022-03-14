import numpy as np
import pandas as pd
from meshparty import meshwork

import sys
from pathlib import Path

path = Path(__file__)
sys.path.append(str(path.absolute().parent))

from apical_classification.src.apical_features import *
from apical_classification.src.apical_model_utils import *
import json

anno_mask_dict = {
    "dendrite": "is_dendrite",
    "basal": "is_basal",
    "apical": "is_apical",
    "soma": "is_soma",
}


def annotate_apical_from_syn_df(nrn, syn_df):
    "Use pre-classified synapse labels to infer skeleton labels"
    apical_syn_df = syn_df.query("is_apical == True")
    if len(apical_syn_df) == 0:
        nrn.anno.add_annotations(anno_mask_dict["apical"], [], mask=True)
        return

    child_verts = nrn.child_index(nrn.root)
    child_is_apical = []
    for vert in child_verts:
        child_is_apical.append(
            np.any(np.isin(nrn.downstream_of(vert), apical_syn_df["post_pt_mesh_ind"]))
        )

    mask_stack = []
    for vert in child_verts[child_is_apical]:
        mask_stack.append(nrn.downstream_of(vert).to_mesh_mask)

    apical_mask = np.any(np.vstack(mask_stack), axis=0)
    nrn.anno.add_annotations(
        anno_mask_dict["apical"], np.flatnonzero(apical_mask), mask=True
    )


def additional_component_masks(nrn, peel_threshold=0.1):
    "Apply soma, basal dendrite, and generic dendrite masks to a neuron"

    apply_dendrite_mask(nrn)
    if peel_threshold is not None:
        peel_sparse_segments(nrn, 0.1)
    dend_mask = nrn.mesh_mask.copy()
    nrn.reset_mask()

    basal_mask = dend_mask.copy()
    basal_mask[nrn.root_region.to_mesh_mask] = False
    basal_mask[nrn.anno.is_apical.mesh_index] = False

    nrn.anno.add_annotations(
        anno_mask_dict["dendrite"], np.flatnonzero(dend_mask), mask=True
    )
    nrn.anno.add_annotations(
        anno_mask_dict["basal"], np.flatnonzero(basal_mask), mask=True
    )
    nrn.anno.add_annotations(anno_mask_dict["soma"], nrn.root_region, mask=True)


def load_root_id(oid, project_paths, syn_dir=None, peel_threshold=0.1):
    "Load and apply dendrite labels to a neuron based on pre-classified synapse apical labels and more"
    if syn_dir is None:
        syn_dir = f"{project_paths.data}/synapse_files/v3"
    nrn = meshwork.load_meshwork(f"{project_paths.skeletons}/skeleton_files/{oid}.h5")
    syn_df = pd.read_feather(f"{syn_dir}/{oid}_inputs.feather")
    annotate_apical_from_syn_df(nrn, syn_df)
    additional_component_masks(nrn, peel_threshold=peel_threshold)
    return nrn


def load_features(root_ids, feature_dir):
    "Load a feature dataframe from a directory"
    dats = []
    for root_id in root_ids:
        try:
            with open(f"{feature_dir}/{root_id}.json") as f:
                dats.append(json.load(f))
        except:
            continue
    return pd.DataFrame.from_records(dats)
