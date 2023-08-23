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
    x_soma_rot, y_soma_rot, z_soma_rot = proper_rotation(
        slice_angle, upright_angle, x_soma, y_soma, z_soma, shrink_factor
    )
    eucl_distance = []
    rot_eucl_distance = []
    for idx in morph_df.index:
        x_node = morph_df.loc[idx, "x"]
        y_node = morph_df.loc[idx, "y"]
        z_node = morph_df.loc[idx, "z"]
        eucl_distance.append(
            sqrt(
                (x_node - x_soma) ** 2 + (y_node - y_soma) ** 2 + (z_node - z_soma) ** 2
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

    max_eucl_distance = max(eucl_distance)
    max_rot_eucl_distance = max(rot_eucl_distance)

    return max_eucl_distance, max_rot_eucl_distance


##########################################################################################################
# VARIABLE PARAMETERS

download_recos = False

cell_id = 479010903

layer = "5"  # '1', '2', '2/3', '3', '4', '6', '6a', '6b'

spex = "Mus musculus"  # 'Homo Sapiens'

neur_type = "spiny"  # 'aspiny'

slice_angle = 3.24431361952

upright_angle = 301.125094376879

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


################################################################################################################################
# VISUALIZATION

# This is the plot of the "shape" of a asingle neuron
fig, ax = plt.subplots(1, 1)

for d_type, color in [
    [2, axon_color],
    [3, bas_dendrite_color],
    [4, api_dendrite_color],
]:
    df = morph_df[morph_df["type"] == d_type]
    ax.scatter(df["x"], df["y"], color=color)
ax.invert_yaxis()
plt.ylabel("y")
plt.xlabel("x")
plt.legend(["axons", "basal dendrites", "apical dendrites"])
# plt.show()

# #This is the plot of cortical depth (of a single neuron)  IT IS NOT BECAUSE WE HAVE TO ROTATE THE NEURON FIRST
# fig,ax=plt.subplots(1,1)
# plt.hist(morph_df.y, bins=6, orientation="horizontal", color = 'm')
# ax.invert_yaxis()
# plt.ylabel("depth")
# plt.xlabel("number")
# #plt.show()

# #This is the normalised plot of cortical depth (of a single neuron)
# data = morph_df.y
# fig,ax=plt.subplots(1,1)
# plt.hist(data/np.max(data), bins=[0,0.1,0.4,0.5,0.8,1.0], orientation="horizontal", color = 'g')
# ax.invert_yaxis()
# plt.ylabel("normalised depth")
# plt.xlabel("number")
# #plt.show()

# This is the plot of the mice_spiny's somas position, colours are associated to layers
fig, ax = plt.subplots(1, 1)
col_dict = {
    "1": "r",
    "2": "#ff7f0e",
    "2/3": "y",
    "3": "g",
    "4": "c",
    "5": "b",
    "6": "#9467bd",
    "6a": "#e377c2",
    "6b": "#7f7f7f",
}
soma_y_coord_vec = []

for cell_idx in mice_spiny_cells_idx:
    l_type = cell_feat_df.loc[cell_idx, "structure__layer"]
    color = col_dict[l_type]
    cell_id = cell_feat_df.loc[cell_idx, "specimen_id"]
    morph = reconstruct(cell_id)
    morph_df = pd.DataFrame(morph.compartment_list)
    soma_x_coord = morph_df.loc[0, "x"]
    soma_y_coord = morph_df.loc[0, "y"]
    soma_y_coord_vec.append(soma_y_coord)
    ax.scatter(soma_x_coord, soma_y_coord, color=color, label=color)

red = mpatches.Patch(color="red", label="Layer 1")
orange = mpatches.Patch(color="#ff7f0e", label="Layer 2")
yellow = mpatches.Patch(color="y", label="Layer 2/3")
green = mpatches.Patch(color="g", label="Layer 3")
cian = mpatches.Patch(color="c", label="Layer 4")
blue = mpatches.Patch(color="b", label="Layer 5")
purple = mpatches.Patch(color="#9467bd", label="Layer 6")
pink = mpatches.Patch(color="#e377c2", label="Layer 6a")
grey = mpatches.Patch(color="#7f7f7f", label="Layer 6b")
ax.invert_yaxis()
plt.ylabel("y")
plt.xlabel("x")
plt.legend(handles=[red, orange, yellow, green, cian, blue, purple, pink, grey])
# plt.show()


# This is an histogram of the y-coordinates of the mice_spiny_cells' somas
fig, ax = plt.subplots(1, 1)
plt.hist(soma_y_coord_vec, bins=6, orientation="horizontal", color=color)
ax.invert_yaxis()
# plt.show()


# This is in an histogram of the overall depth of mice_spiny_cells, colours are associated to layers
fig, ax = plt.subplots(1, 1)
for cell_idx in mice_spiny_cells_idx:
    l_type = cell_feat_df.loc[cell_idx, "structure__layer"]
    color = col_dict[l_type]
    depth = cell_feat_df.loc[cell_idx, "overall_depth"]
    ax.hist(depth, orientation="horizontal", color=color, label=color)

red = mpatches.Patch(color="red", label="Layer 1")
orange = mpatches.Patch(color="#ff7f0e", label="Layer 2")
yellow = mpatches.Patch(color="y", label="Layer 2/3")
green = mpatches.Patch(color="g", label="Layer 3")
cian = mpatches.Patch(color="c", label="Layer 4")
blue = mpatches.Patch(color="b", label="Layer 5")
purple = mpatches.Patch(color="#9467bd", label="Layer 6")
pink = mpatches.Patch(color="#e377c2", label="Layer 6a")
grey = mpatches.Patch(color="#7f7f7f", label="Layer 6b")
ax.invert_yaxis()
plt.ylabel("y")
plt.xlabel("x")
plt.legend(handles=[red, orange, yellow, green, cian, blue, purple, pink, grey])
# plt.show()


# This is an histogram of the soma distance form pia for the mice_spiny cellls that has also orientation data (same colour so no distinction between layers)
fig, ax = plt.subplots(1, 1)
plt.hist(
    cell_feat_orient_new_df.soma_distance_from_pia, orientation="horizontal", color="m"
)
ax.invert_yaxis()
plt.ylabel("depth")
plt.xlabel("number")
# plt.show()


depth = []
# This is a histogram of the soma distance form pia for the mice_spiny cellls that has also orientation data (this time with distinction of layers that depends on colours)
fig, ax = plt.subplots(1, 1)
for cell_idx in mice_spiny_orient_cells_idx:
    l_type = cell_feat_orient_new_df.loc[cell_idx, "structure__layer"]
    color = col_dict[l_type]
    # if cell_feat_orient_df.loc[cell_idx,'soma_distance_from_pia'] != np.nan:
    #       depth.append(cell_feat_orient_df.loc[cell_idx,'soma_distance_from_pia'])
    #       ax.hist(depth[-1], orientation="horizontal", color = color, label = color)
    depth = cell_feat_orient_new_df.loc[cell_idx, "soma_distance_from_pia"]
    ax.hist(depth, orientation="horizontal", color=color, label=color)
red = mpatches.Patch(color="red", label="Layer 1")
orange = mpatches.Patch(color="#ff7f0e", label="Layer 2")
yellow = mpatches.Patch(color="y", label="Layer 2/3")
green = mpatches.Patch(color="g", label="Layer 3")
cian = mpatches.Patch(color="c", label="Layer 4")
blue = mpatches.Patch(color="b", label="Layer 5")
purple = mpatches.Patch(color="#9467bd", label="Layer 6")
pink = mpatches.Patch(color="#e377c2", label="Layer 6a")
grey = mpatches.Patch(color="#7f7f7f", label="Layer 6b")
ax.invert_yaxis()
plt.ylabel("soma_depth")
plt.xlabel("x")
plt.legend(handles=[red, orange, yellow, green, cian, blue, purple, pink, grey])
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for d_type, color in [
    [2, axon_color],
    [3, bas_dendrite_color],
    [4, api_dendrite_color],
]:
    df = morph_df[morph_df["type"] == d_type]
    x_coord, y_coord, z_coord = proper_rotation(
        slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
    )
    ax.scatter(x_coord, z_coord, y_coord, color=color)
ax.invert_xaxis()
plt.ylabel("y")
plt.xlabel("x")
plt.legend(["axons", "basal dendrites", "apical dendrites"])
# plt.show()


fig, ax = plt.subplots(1, 1)
for d_type, color in [
    [2, axon_color],
    [3, bas_dendrite_color],
    [4, api_dendrite_color],
]:
    df = morph_df[morph_df["type"] == d_type]
    x_coord, y_coord, z_coord = proper_rotation(
        slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
    )
    ax.scatter(x_coord, y_coord, color=color)
ax.invert_yaxis()
plt.ylabel("y")
plt.xlabel("x")
plt.legend(["axons", "basal dendrites", "apical dendrites"])
plt.show()
