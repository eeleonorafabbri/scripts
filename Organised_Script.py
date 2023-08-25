# Organised Script

# IMPORTS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.core.cell_types_cache import ReporterStatus as RS

from Viz import Viz

# builtins
from pathlib import Path
import time
import pprint
import pdb
from copy import deepcopy
import math
from math import sqrt

# GLOBAL PARAMETERS

output_dir = Path("/opt3/Eleonora/data/reconstruction")
ctc = CellTypesCache(manifest_file=output_dir / "manifest.json")
axon_color = "blue"
bas_dendrite_color = "red"
api_dendrite_color = "orange"


# FUNCTIONS


def do_it_all():
    feat_df = ctc.get_all_features(dataframe=True)
    ef_df = pd.read_csv("/opt3/Eleonora/data/ef_data.csv")
    mor_df = pd.read_csv("/opt3/Eleonora/data/mor_data.csv")
    return ef_df, mor_df, feat_df


def reconstruct(id_cell):
    morphology = ctc.get_reconstruction(id_cell)
    return morphology


def which_layer(layer, cell_feat_df):
    cells_in_layer = cell_feat_df[
        cell_feat_df["structure__layer"].values == layer
    ].index
    return cells_in_layer


def spiny_or_aspiny_cells(cell_feat_df):
    spiny_cells = cell_feat_df[
        cell_feat_df["tag__dendrite_type"].values == "spiny"
    ].index
    aspiny_cells = cell_feat_df[
        cell_feat_df["tag__dendrite_type"].values == "aspiny"
    ].index
    return spiny_cells, aspiny_cells


def species(cell_feat_df):
    human_cells = cell_feat_df[
        cell_feat_df["donor__species"].values == "Homo Sapiens"
    ].index
    mice_cells = cell_feat_df[
        cell_feat_df["donor__species"].values == "Mus musculus"
    ].index
    return human_cells, mice_cells


def layer_type_species(layer, spex, neur_type, cell_feat_df):
    cells_in_layer = which_layer(layer, cell_feat_df)
    if spex == "Homo Sapiens":
        spex_cells = human_cells
    if spex == "Mus musculus":
        spex_cells = mice_cells
    if neur_type == "spiny":
        type_cells = spiny_cells
    if neur_type == "aspiny":
        type_cells = aspiny_cells
    layer_type_spex = set(cells_in_layer) & set(type_cells) & set(spex_cells)
    layer_type_spex_idx = np.array(list(layer_type_spex))
    layer_type_spex_idx.sort()
    layer_type_spex_idx_df = cell_feat_df.loc[layer_type_spex_idx]
    return layer_type_spex_idx_df


def axon_or_dendrite(morph_df):
    axons_idx = morph_df[morph_df["type"].values == 2].index
    basal_dendrite_idx = morph_df[morph_df["type"].values == 3].index
    apical_dendrite_idx = morph_df[morph_df["type"].values == 4].index
    return axons_idx, basal_dendrite_idx, apical_dendrite_idx


def correct_slice_angle(alpha, x, y):
    alpha = math.radians(alpha)
    x_new = x * (math.cos(alpha)) + y * (math.sin(alpha))
    y_new = -x * (math.sin(alpha)) + y * (math.cos(alpha))
    return x_new, y_new


def proper_rotation(slice_angle, upright_angle, x1, y1, z1, shrink_factor):
    slice_angle = math.radians(slice_angle)
    upright_angle = math.radians(upright_angle)
    z1 = z1 * shrink_factor
    x2 = x1 * (math.cos(upright_angle)) - y1 * (math.sin(upright_angle))
    y2 = x1 * (math.sin(upright_angle)) + y1 * (math.cos(upright_angle))
    z2 = z1
    x3 = x2
    y3 = y2 * (math.cos(slice_angle)) - z2 * (math.sin(slice_angle))
    z3 = y2 * (math.sin(slice_angle)) + z2 * (math.cos(slice_angle))
    return x3, y3, z3


def find_depth(slice_angle, upright_angle, shrink_fac, x, y, z):
    z = z * shrink_fac
    x_coord, y_coord, z_coord = proper_rotation(slice_angle, upright_angle, x, y, z)
    depth = sqrt(x_coord[0] ** 2 + y_coord[0] ** 2 + z_coord[0] ** 2)
    return depth


def find_max_eucl_distance(cell_id, cell_feat_orient_new_df):
    morph = reconstruct(cell_id)
    morph_df = pd.DataFrame(morph.compartment_list)
    x_soma = morph_df.loc[0, "x"]
    y_soma = morph_df.loc[0, "y"]
    z_soma = morph_df.loc[0, "z"]
    cell_idx = cell_feat_orient_new_df[
        cell_feat_orient_new_df["specimen_id"] == cell_id
    ].index
    slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
    upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
    shrink_factor = cell_feat_orient_new_df.loc[cell_idx, "estimated_shrinkage_factor"]
    z_soma_shrink = z_soma * shrink_factor
    x_soma_rot, y_soma_rot, z_soma_rot = proper_rotation(
        slice_angle, upright_angle, x_soma, y_soma, z_soma, shrink_factor
    )
    eucl_distance = []
    shrinked_eucl_distance = []
    rot_eucl_distance = []
    xy_distance = []
    xy_rot_distance = []
    for idx in morph_df.index:
        x_node = morph_df.loc[idx, "x"]
        y_node = morph_df.loc[idx, "y"]
        z_node = morph_df.loc[idx, "z"]
        eucl_distance.append(
            sqrt(
                (x_node - x_soma) ** 2 + (y_node - y_soma) ** 2 + (z_node - z_soma) ** 2
            )
        )
        z_node_shrink = z_node * shrink_factor
        shrinked_eucl_distance.append(
            sqrt(
                (x_node - x_soma) ** 2
                + (y_node - y_soma) ** 2
                + (z_node_shrink - z_soma_shrink) ** 2
            )
        )
        x_node_rot, y_node_rot, z_node_rot = proper_rotation(
            slice_angle, upright_angle, x_node, y_node, z_node, shrink_factor
        )
        rot_eucl_distance.append(
            sqrt(
                (x_node_rot - x_soma_rot) ** 2
                + (y_node_rot - y_soma_rot) ** 2
                + (z_node_rot - z_soma_rot) ** 2
            )
        )
        xy_distance.append(sqrt((x_node - x_soma) ** 2 + (y_node - y_soma) ** 2))
        xy_rot_distance.append(
            sqrt((x_node_rot - x_soma_rot) ** 2 + (y_node_rot - y_soma_rot) ** 2)
        )

    max_eucl_distance = max(eucl_distance)
    max_shrinked_eucl_distance = max(shrinked_eucl_distance)
    max_rot_eucl_distance = max(rot_eucl_distance)
    max_xy_distance = max(xy_distance)
    max_xy_rot_distance = max(xy_rot_distance)
    # idx_position_far_node = xy_rot_distance.index(max_xy_rot_distance)

    return (
        max_eucl_distance,
        max_rot_eucl_distance,
        max_shrinked_eucl_distance,
        max_xy_distance,
        max_xy_rot_distance,
        # idx_position_far_node,
    )


def calc_distance_from_pia(cell_id, idx_node, cell_feat_orient_new_df):
    morph = reconstruct(cell_id)
    morph_df = pd.DataFrame(morph.compartment_list)
    cell_idx = cell_feat_orient_new_df[
        cell_feat_orient_new_df["specimen_id"] == cell_id
    ].index
    soma_distance_from_pia = cell_feat_orient_new_df.loc[
        cell_idx, "soma_distance_from_pia"
    ]
    slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
    upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
    shrink_factor = cell_feat_orient_new_df.loc[cell_idx, "estimated_shrinkage_factor"]
    x_soma, y_soma, z_soma = proper_rotation(
        slice_angle,
        upright_angle,
        morph_df.loc[0, "x"],
        morph_df.loc[0, "y"],
        morph_df.loc[0, "z"],
        shrink_factor,
    )
    x_far_node, y_far_node, z_far_node = proper_rotation(
        slice_angle,
        upright_angle,
        morph_df.loc[idx_node, "x"],
        morph_df.loc[idx_node, "y"],
        morph_df.loc[idx_node, "z"],
        shrink_factor,
    )
    # x_pia = x_soma
    y_pia = y_soma + soma_distance_from_pia
    # z_pia = z_soma  # I don't know if this is correct
    # distance_from_pia = sqrt(
    #     (x_far_node - x_pia) ** 2
    #     + (y_far_node - y_pia) ** 2
    #     + (z_far_node - z_pia) ** 2
    # )
    distance_from_pia = y_pia - y_far_node
    return distance_from_pia


##########################################################################################################
# VARIABLE PARAMETERS

download_recos = False

cell_id = 479013100

layer = "2/3"  # '1', '2', '3', '4', '5', '6', '6a', '6b'

spex = "Mus musculus"  # 'Homo Sapiens'

neur_type = "spiny"  # 'aspiny'

slice_angle = 0

upright_angle = 176.909187120959

shrink_factor = 3.05757172357999

##########################################################################################################
# COMPUTATION W VARIABLE PARAMETERS

ef_df, mor_df, feat_df = do_it_all()


cell_feat_df = deepcopy(feat_df)
needed_columns = [
    "specimen__id",
    "specimen__name",
    "specimen__hemisphere",
    "structure__id",
    "structure__name",
    "structure__acronym",
    "structure_parent__id",
    "structure_parent__acronym",
    "structure__layer",
    "nr__average_parent_daughter_ratio",
    "nrwkf__id",
    "erwkf__id",
    "ef__avg_firing_rate",
    "si__height",
    "si__width",
    "si__path",
    "csl__x",
    "csl__y",
    "csl__z",
    "csl__normalized_depth",
    "cell_reporter_status",
    "m__glif",
    "m__biophys",
    "m__biophys_perisomatic",
    "m__biophys_all_active",
    "tag__apical",
    "tag__dendrite_type",
    "morph_thumb_path",
    "ephys_thumb_path",
    "ephys_inst_thresh_thumb_path",
    "donor__age",
    "donor__sex",
    "donor__disease_state",
    "donor__race",
    "donor__years_of_seizure_history",
    "donor__species",
    "donor__id",
    "donor__name",
]

cell_feat_df = cell_feat_df.reindex(
    columns=cell_feat_df.columns.tolist() + needed_columns
)

not_needed_columns = [
    "nr__number_bifurcations",
    "nr__average_contraction",
    "nr__reconstruction_type",
    "nr__max_euclidean_distance",
    "nr__number_stems",
    "ef__fast_trough_v_long_square",
    "ef__upstroke_downstroke_ratio_long_square",
    "ef__adaptation",
    "ef__f_i_curve_slope",
    "ef__threshold_i_long_square",
    "ef__tau",
    "ef__avg_isi",
    "ef__ri",
    "ef__peak_t_ramp",
    "ef__vrest",
    "line_name",
]


if download_recos == True:
    for this_row in feat_df.index:
        try:
            file_name = f'specimen_id_{feat_df.loc[this_row,"specimen_id"]}'
            full_filename = output_dir / file_name
            print(full_filename)
            ex = ctc.get_reconstruction(
                feat_df.loc[this_row, "specimen_id"], file_name=full_filename
            )
        except Exception:
            print(
                f'Reco not found for cell {feat_df.loc[this_row,"specimen_id"]} at row={this_row}'
            )

cell_df = pd.read_csv("/opt3/Eleonora/data/cell_types_specimen_details_3.csv")

morph = reconstruct(cell_id)
morph_df = pd.DataFrame(morph.compartment_list)

common_id = set(cell_df.specimen__id) & set(feat_df.specimen_id)

for cell_id in common_id:
    id_row = cell_df.loc[cell_df["specimen__id"] == cell_id, needed_columns]
    specimen__id = id_row["specimen__id"].values
    row_index = cell_feat_df[cell_feat_df["specimen_id"].values == specimen__id].index
    cell_feat_df.loc[row_index, needed_columns] = id_row.values

cells_in_layer = which_layer(layer, cell_feat_df)

spiny_cells, aspiny_cells = spiny_or_aspiny_cells(cell_feat_df)

human_cells, mice_cells = species(cell_feat_df)

layer_type_spex_idx_df = layer_type_species(layer, spex, neur_type, cell_feat_df)

axons_idx, basal_dendrite_idx, apical_dendrite_idx = axon_or_dendrite(morph_df)

mice_spiny_cells_idx = set(mice_cells) & set(spiny_cells)

orientation_df = pd.read_csv("/opt3/Eleonora/data/orientation_data.csv")
orient_id = set(orientation_df.specimen_id) & set(cell_feat_df.specimen_id)


cell_feat_orient_df = deepcopy(cell_feat_df)

cell_feat_orient_df[list(orientation_df.columns)] = pd.DataFrame(
    [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
    index=cell_feat_orient_df.index,
)

for cell_id in orient_id:
    id_row = orientation_df.loc[
        orientation_df["specimen_id"] == cell_id, list(orientation_df.columns)
    ]
    specimen__id = id_row["specimen_id"].values
    row_index = cell_feat_orient_df[cell_feat_df["specimen_id"].values == cell_id].index
    cell_feat_orient_df.loc[row_index, list(orientation_df.columns)] = id_row.values


cell_feat_orient_new_df = cell_feat_orient_df.dropna(
    subset=["soma_distance_from_pia"]
)  # It's the dataframe that contains only cata of cells whose soma_distance_from_pia is not nan

spiny_orient_cells, aspiny_orient_cells = spiny_or_aspiny_cells(cell_feat_orient_new_df)

human_orient_cells, mice_orient_cells = species(cell_feat_orient_new_df)

mice_spiny_orient_cells_idx = set(spiny_orient_cells) & set(mice_orient_cells)


# x_ax = []
# y_ax = []
# x_bas_den = []
# y_bas_den = []
# x_api_den = []
# y_api_den = []

# for idx in axons_idx:
#     x_ax.append(morph_df.loc[idx,'x'].tolist())
#     y_ax.append(morph_df.loc[idx,'y'].tolist())

# for idx in basal_dendrite_idx:
#     x_bas_den.append(morph_df.loc[idx,'x'].tolist())
#     y_bas_den.append(morph_df.loc[idx,'y'].tolist())

# for idx in apical_dendrite_idx:
#     x_api_den.append(morph_df.loc[idx,'x'].tolist())
#     y_api_den.append(morph_df.loc[idx,'y'].tolist())

cell_id1 = 488117124
(
    max_xyz_distance1,
    max_xyz_rot_distance1,
    max_shrinked_eucl_distance1,
    max_xy_distance1,
    max_xy_rot_distance1,
) = find_max_eucl_distance(cell_id1, cell_feat_orient_new_df)
cell_idx = cell_feat_orient_new_df[
    cell_feat_orient_new_df["specimen_id"] == cell_id1
].index
soma_dist_xyz = cell_feat_orient_new_df.loc[cell_idx, "soma_distance_from_pia"]
# soma_dist_xy = (max_xy_distance1 * soma_dist_xyz) / max_xyz_distance1
# soma_dist_xy = (max_xy_distance1 * soma_dist_xyz) / max_shrinked_eucl_distance1
soma_dist_xy = (max_xy_rot_distance1 * soma_dist_xyz) / max_xyz_rot_distance1


specimens = [
    479013100,
    567952169,  # not problematic
    582644266,
    501799874,
    497611660,
    535708196,
    586073683,
    478888052,
    487664663,
    554807924,
    478499902,
    478586295,
    510715606,
    569723367,
    485837504,
    479010903,
    471141261,
    314900022,
    512322162,
    # 313862167,313862167,
    585832440,
    502999078,
    573404307,
    476049169,
    480351780,
    580162374,
    386049446,
    397353539,
    475585413,  # PROBLEMATIC !!!!!
    501845152,  # not problematic
    329550137,  # not problematic
    488117124,
    574067499,
    486560376,  # not problematic
    485184849,
    567354967,
    591268268,
    # 478110866,478110866,
    485835016,
    589760138,
    480114344,
    530737765,
    515524026,
    583146274,
    562541627,
    574734127,
    476616076,
    # 565417112,565417112,
    333785962,
    476048909,
    471087830,
    585952606,
    524689239,  # not problematic
    476451456,
    471767045,
    # 321708130,321708130,
    480003970,
    480116737,
    483020137,
    515986607,
    594091004,
    321906005,
    565863515,
    569723405,
    609435731,
    515249852,
    422738880,
    487601493,
    471786879,
    580010290,
    # 473540161,473540161,
    480124551,
    579662957,  # not problematic
    555345752,
    476126528,
    478892782,
    584231995,
    557037024,
    # 556968207,556968207,
    486111903,
    582917630,  # not problematic
    488501071,
    475202388,
    580161872,
    585947309,  # not problematic
    475068599,
    519749342,
    510658021,
    485835055,
    586071425,
    561517025,
    476131588,
    471077857,
    584872371,
    584680861,
]

# This takes like 10 seconds for doing it for 100 cells
for cell_id in specimens:
    morph = reconstruct(cell_id)
    morph_df = pd.DataFrame(morph.compartment_list)
    cell_idx = cell_feat_orient_new_df[
        cell_feat_orient_new_df["specimen_id"] == cell_id
    ].index
    slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"].values
    upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"].values
    shrink_factor = cell_feat_orient_new_df.loc[
        cell_idx, "estimated_shrinkage_factor"
    ].values
    x_coord, y_coord, z_coord = proper_rotation(
        slice_angle,
        upright_angle,
        morph_df["x"],
        morph_df["y"],
        morph_df["z"],
        shrink_factor,
    )
    morph_df["x"] = x_coord
    morph_df["y"] = y_coord
    morph_df["z"] = z_coord


################################################################################################################################
# VISUALIZATION

viz = Viz()
