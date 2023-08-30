# Organised Script

# IMPORTS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve, newton, bisect

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
    """
    This function returns the indices of the cells in the layer specified as input.
    """
    cells_in_layer = cell_feat_df[
        cell_feat_df["structure__layer"].values == layer
    ].index
    return cells_in_layer


def spiny_or_aspiny_cells(cell_feat_df):
    """
    This function returns the indices of the spiny and aspiny cells.
    """
    spiny_cells = cell_feat_df[
        cell_feat_df["tag__dendrite_type"].values == "spiny"
    ].index
    aspiny_cells = cell_feat_df[
        cell_feat_df["tag__dendrite_type"].values == "aspiny"
    ].index
    return spiny_cells, aspiny_cells


def species(cell_feat_df):
    """
    This function returns the indices of the human and mice cells.
    """
    human_cells = cell_feat_df[
        cell_feat_df["donor__species"].values == "Homo Sapiens"
    ].index
    mice_cells = cell_feat_df[
        cell_feat_df["donor__species"].values == "Mus musculus"
    ].index
    return human_cells, mice_cells


def layer_type_species(layer, spex, neur_type, cell_feat_df):
    """
    This function returns the dtaframe containing the cells with conditions in input on layers,
    species and spiny/aspiny features
    """
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
    """
    This function returns the index of axons, basal_dendrites and apical_dendrites
    """
    axons_idx = morph_df[morph_df["type"].values == 2].index
    basal_dendrite_idx = morph_df[morph_df["type"].values == 3].index
    apical_dendrite_idx = morph_df[morph_df["type"].values == 4].index
    return axons_idx, basal_dendrite_idx, apical_dendrite_idx


def correct_slice_angle(alpha, x, y):
    """
    This function is a simple rotation in 2D, around the origo, of an angle alpha
    """
    alpha = math.radians(alpha)
    x_new = x * (math.cos(alpha)) + y * (math.sin(alpha))
    y_new = -x * (math.sin(alpha)) + y * (math.cos(alpha))
    return x_new, y_new


def proper_rotation(slice_angle, upright_angle, x1, y1, z1, shrink_factor):
    """
    This function is a rotation in 3D of an angle slice_angle around the x axis,
    followed by a rotation of an angle upright_angle around the z axis.
    Before the rotation the z coordinate is shrinked by a factor shrink_factor.
    """
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


def _find_depth(slice_angle, upright_angle, shrink_fac, x, y, z):
    z = z * shrink_fac
    x_coord, y_coord, z_coord = proper_rotation(slice_angle, upright_angle, x, y, z)
    depth = sqrt(x_coord[0] ** 2 + y_coord[0] ** 2 + z_coord[0] ** 2)
    return depth


def find_max_eucl_distance(cell_id, cell_feat_orient_new_df):
    """
    This function returns the maximum euclidean distance between the soma and the farthest node of the cell,
    the maximum euclidean distance between the soma and the farthest node of the cell after the rotation,
    the maximum euclidean distance between the soma and the farthest node of the cell after the rotation and
    the shrinkage (which is identical to the one after the rotation),the maximum distance between the soma
    and the farthest node of the cell in the xy plane, the maximum distance between the soma and the farthest
    node of the cell after the rotation in the xy plane, and the index of the farthest node of the cell after
    the rotation in the xy plane."""

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
    """
    This function returns the distance from the pia of the node with index idx_node of
    the cell with id cell_id.
    """

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
    x_node, y_node, z_node = proper_rotation(
        slice_angle,
        upright_angle,
        morph_df.loc[idx_node, "x"],
        morph_df.loc[idx_node, "y"],
        morph_df.loc[idx_node, "z"],
        shrink_factor,
    )
    y_pia = y_soma + soma_distance_from_pia
    distance_from_pia = y_pia - y_node
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

VISp_cells_idx = cell_feat_orient_new_df[
    cell_feat_orient_new_df["structure_parent__acronym"].values == "VISp"
].index

VISp_mice_cells_idx = set(mice_orient_cells) & set(VISp_cells_idx)


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

cell_id1 = 501799874

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
soma_dist_xy = (max_xy_rot_distance1 * soma_dist_xyz) / max_xyz_rot_distance1


specimens = [
    479013100,
    # 567952169,  # not problematic
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
    # 475585413,  # PROBLEMATIC !!!!!
    # 501845152,  # not problematic
    # 329550137,  # not problematic
    488117124,
    574067499,
    # 486560376,  # not problematic
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
    # 524689239,  # not problematic
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
    # 579662957,  # not problematic
    555345752,
    476126528,
    478892782,
    584231995,
    557037024,
    # 556968207,556968207,
    486111903,
    # 582917630,  # not problematic
    488501071,
    475202388,
    580161872,
    # 585947309,  # not problematic
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

specimens2 = [
    585944237,
    469798159,
    502359001,
    # 515771244,  # not problematic
    484679812,
    486110216,
    563226105,
    479770916,
    # 585841870,  # not problematic
    313862373,
    354190013,
    324025371,
    485931158,
    536951541,
    485912047,
    323865917,
    # 555341581,
    # 555341581,
    570896413,
    571311039,
    # 521409057,  # PROBLEMATIC !!!!!
    526531616,
    560678143,
    341442651,
    475744706,
    468193142,
    # 565866518,
    # 565866518,
    561985849,
    577369606,
    502269786,
    483061182,
    602822298,
    567007144,
    313862022,
    554779051,
    607124114,
    565855793,
    487661754,
    # 488680211,  # not problematic
    # 396608557,  # not problematic
    490205998,
    483068687,
    563180078,
    574993444,
    515435668,
    517319635,
    565880475,
    561940338,
    560753350,
    # 476823462,  # not problematic
    # 479704527,  # not problematic
    502978383,
    # 562632795,
    # 562632795,
    561325425,
    480169178,
    574036994,
    # 466632464,  # not problematic
    578485753,
    382982932,
    # 488677994,
    # 488677994,
    473020156,
    488687894,
    586379590,
    # 523748610,
    # 523748610,
    592952680,
    314642645,
    527826878,
    510136749,
    486146717,
    512319604,
    562003142,
    567312865,
    517077548,
    # 555001065,  # not problematic
    466245544,
    479728896,
    571379222,
    # 569072334,
    # 569072334,
    # 574038330,  # not problematic
    585925172,
    485574832,
    473601979,
    554454458,
    395830185,
    486893033,
    530731648,
    # 478058328,  # not problematic
    571396942,
    488697163,
    490387590,
    477880128,
    325941643,
    509515969,
    469793303,
    575642695,
]

specimens3 = [
    490259231,
    # 555089724,
    # 555089724,
    584254833,
    598628992,
    485909730,
    488698341,
    479905853,
    589442285,
    476054887,
    571306690,
    # 535728342,  # not problematic
    476455864,
    589427435,
    483101699,
    585830272,
    # 573622646,  # not problematic
    488680917,
    509003464,
    578774163,
    # 509617624,
    # 509617624,
    580005568,
    486262299,
    318733871,
    515464483,
    # 570946690,  # not problematic
    354833767,
    # 475549284,
    # 475549284,
    # 476086391,
    # 476086391,
    534303031,
    583138568,
    # 471410185,  # not problematic
    514767977,
    479225052,
    324493977,
    # 527095729,  # PROBLEMATIC !!!!!
    560965993,
    586072188,
    485161419,
    490916882,
    599334696,
    555241875,
    565888394,
    476562817,
    488504814,
    571867358,
    # 485836906,  # not problematic
    329550277,
    348592897,
    579626068,
    # 487099387,  # PROBLEMATIC !!!!! (but also for allen)
    480087928,
    583104750,
    614777438,
    565462089,
    473943881,
    539014038,
    586072464,
    469992918,
    468120757,
    469704261,
    564349611,
    479179020,
    # 572609108,  # not problematic
    # 314831019,  # not problematic
    557874460,
    488695444,
    555241040,
    555019563,
    586566174,
    589128331,
    569809287,
    580145037,
    # 486896849,
    # 486896849,
    570896453,
    320668879,
    561532710,
    397351623,
    526668864,
    566647353,
    323452196,
    490485142,
    569998790,
    # 583138230,  # not problematic
    473543792,
    473564515,
    485468180,
    324025297,
    564346637,
    565209132,
    475459689,
    474626527,
    566761793,
    514824979,
    565120091,
    501845630,
]

specimens4 = [
    623185845,
    570438732,
    386970660,
    522301604,
    502366405,
    603402458,
    566678900,
    590558808,
    565407476,
    486025194,
    # 479721491,
    # 479721491,
    526643573,
    471143169,
    323452245,
    471129934,
    481093525,
    476218657,
    483092310,
    574059157,
    314804042,
    476686112,
    577218602,
    566541912,
    324266189,
    471077468,
    557261437,
    557998843,
    513510227,
    582613001,
    603423462,
    482516713,
    327962063,
    # 323475862,
    # 323475862,
    506133241,
    477135941,
    547344442,
    547325858,
    584682764,
    # 569810649,
    # 569810649,
    585951863,
    476216637,
    508980706,
    488679042,
    568568071,
    502267531,
    333604946,
    517345160,
    488683425,
    # 536306115,
    # 536306115,
    586073850,
    560992553,
    567029686,
    574377552,
    # 561934585,
    # 561934585,
    567320213,
    527869035,
    522152249,
    517647182,
    314822529,  #
    480353286,
    501841672,
    574992320,
    486132712,
    486239338,
    501850666,
    567927838,
    507101551,
    605660220,
    569958754,
    318808427,
    521938313,
    467703703,
    485574721,  #
    471800189,
    481136138,
    503814817,
    526950199,
    485911892,
    475057898,
    320207387,
    614767057,  #
    480122859,
    588712191,
    473611755,
    471789504,
    580007431,
    476135066,
    515315072,
    513800575,
    507918877,
    513531471,
    557864274,
    585946742,
    583434059,
    575774870,
    559387643,
    509881736,
    565415071,
    588402092,
    # 556923554,
    # 556923554,
    555697303,
    484564503,
    490382353,
    502367941,
    # 534141324,
    # 534141324,
    475623964,
    503286448,
    557252022,
    485838981,
    479225080,
    501847931,
    578938153,
    502383543,
    502614426,
    471758398,
    566671538,
    349621141,
    569670455,
    476263004,
    591278744,
    476457450,
    # 593312584,
    # 593312584,
    488420599,
    527116037,
    # 477127614,
    # 477127614,
    591265629,
    324521027,
    569965244,
    580014328,
    516362762,
    486502127,
    # 569494755,
    # 569494755,
    585805211,
    593321019,
    478497530,
    503823047,
    586559181,
    586072783,
    526573598,
    485880739,
    486052980,
    524850271,
    547262585,
    501956013,
    587045566,
    370351753,
    565459685,
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

viz.plot_in_layer(cell_id, cell_feat_orient_new_df, VISp_mice_cells_idx)
plt.show()
