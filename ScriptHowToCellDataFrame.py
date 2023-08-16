import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.core.cell_types_cache import ReporterStatus as RS

# builtins
from pathlib import Path
import time
import pprint
import pdb
from copy import deepcopy

#This is a script in which I will try to describe how to make DataFrames for Electrophisiological and Morphological cells datas

#EP DATAS
#I just used the Allen resources at the link: https://allensdk.readthedocs.io/en/latest/_static/examples/nb/cell_types.html#Cell-Types-Database and after imported all the CellTypesCache and everything
#I just followed the informations in the Electrophisiological part of the explanation

#MORPH DATAS
#This time it was a bit more complicated because I didn't hadjust to follow the instructions from the link above, but I had to make some changes.
#Afetr importing everything, I had to assigne to a variable (which I called mor_df) the command  >>> mor_df = ctc.get_morphology_features(dataframe=True)
#Doing that allowed us to obtain the DataFrame we wanted

#I saved those DataFrames in ope3/Eleonora/data as ef_data.csv and ef_data.csv

#Indeed I have discovered that we can make a single DataFrame just using the method .get_all_features(dataframe=True) 
#I saved this csv as feat_df in the same place as the previous two

output_dir = Path('/opt3/Eleonora/data/reconstruction')
ctc = CellTypesCache(manifest_file=output_dir / 'manifest.json')


def do_it_all():
    # import pandas as pd
    # import numpy as np
    # from pathlib import Path
    #global ctc
    #cells = ctc.get_cells()
    feat_df = ctc.get_all_features(dataframe=True)
    # global ephys_features
    ephys_features = ctc.get_ephys_features()
    # global ef_df
    ef_df = pd.DataFrame(ephys_features)
    #ef_df = pd.read_csv('/opt3/Eleonora/data/ef_data.csv')
    mor_df = pd.read_csv('mor_data.csv')
    return ephys_features, ef_df,mor_df,feat_df

download_recos = False
ephys_features, ef_df ,mor_df, feat_df= do_it_all()

if download_recos == True:
    for this_row in feat_df.index:
        try: 
            file_name=f'specimen_id_{feat_df.loc[this_row,"specimen_id"]}'
            full_filename = output_dir / file_name
            print(full_filename)
            ex = ctc.get_reconstruction(feat_df.loc[this_row,'specimen_id'], file_name=full_filename)
        except Exception: 
            print(f'Reco not found for cell {feat_df.loc[this_row,"specimen_id"]} at row={this_row}')

cell_df = pd.read_csv('/opt3/Eleonora/data/cell_types_specimen_details_3.csv')

def reconstruct(id_cell): 
    morphology=ctc.get_reconstruction(id_cell)
    # pprint.pprint(morphology.compartment_list[:])
    return morphology

cell_id = 478107198
morph = reconstruct(cell_id)
morph_df = pd.DataFrame(morph.compartment_list)
# ex_type = 3
# cells_type = morph_df[morph_df["type"].values == ex_type].index


common_id = set(cell_df.specimen__id)&set(feat_df.specimen_id)

# for id_cell in common_id:
#     morph = reconstruct(id_cell)
#     morph_df = pd.DataFrame(morph.compartment_list)



cell_feat_df = deepcopy(feat_df)
needed_columns = [ 'specimen__id','specimen__name', 'specimen__hemisphere',
  'structure__id', 'structure__name', 'structure__acronym',
  'structure_parent__id', 'structure_parent__acronym', 'structure__layer',
  'nr__average_parent_daughter_ratio',
  'nrwkf__id', 'erwkf__id', 'ef__avg_firing_rate', 'si__height', 'si__width', 'si__path', 'csl__x', 'csl__y',
  'csl__z', 'csl__normalized_depth', 'cell_reporter_status', 'm__glif',
  'm__biophys', 'm__biophys_perisomatic', 'm__biophys_all_active',
  'tag__apical', 'tag__dendrite_type', 'morph_thumb_path',
  'ephys_thumb_path', 'ephys_inst_thresh_thumb_path', 'donor__age',
  'donor__sex', 'donor__disease_state', 'donor__race',
  'donor__years_of_seizure_history', 'donor__species', 'donor__id',
  'donor__name']

cell_feat_df = cell_feat_df.reindex(columns = cell_feat_df.columns.tolist() + needed_columns)

not_needed_columns = ['nr__number_bifurcations', 'nr__average_contraction','nr__reconstruction_type','nr__max_euclidean_distance',
                    'nr__number_stems','ef__fast_trough_v_long_square','ef__upstroke_downstroke_ratio_long_square', 'ef__adaptation','ef__f_i_curve_slope', 'ef__threshold_i_long_square', 'ef__tau',
                    'ef__avg_isi','ef__ri', 'ef__peak_t_ramp','ef__vrest','line_name']


# id_row = cell_df.loc[cell_df['specimen__id']==cell_id, needed_columns]
# specimen__id = id_row['specimen__id'].values
# row_index = cell_feat_df[cell_feat_df["specimen_id"].values == specimen__id].index
# cell_feat_df.loc[row_index, needed_columns] = id_row.values


#Now we created a DataFrame (cell_feat_df) that has all the features we want, so are included info on layers and type
for cell_id in common_id:
    id_row = cell_df.loc[cell_df['specimen__id']==cell_id, needed_columns]

    specimen__id = id_row['specimen__id'].values

    row_index = cell_feat_df[cell_feat_df["specimen_id"].values == specimen__id].index
    cell_feat_df.loc[row_index, needed_columns] = id_row.values


def which_layer(layer):
    cells_in_layer = cell_feat_df[cell_feat_df["structure__layer"].values == layer].index
    return cells_in_layer

ex_layer = '5' # ...
cells_in_layer = which_layer(ex_layer)

def spiny_or_aspiny_cells():
    spiny_cells = cell_feat_df[cell_feat_df["tag__dendrite_type"].values == 'spiny'].index
    aspiny_cells = cell_feat_df[cell_feat_df["tag__dendrite_type"].values == 'aspiny'].index
    return spiny_cells, aspiny_cells

spiny_cells, aspiny_cells = spiny_or_aspiny_cells()

layer5_spiny = set(cells_in_layer)&set(spiny_cells)
layer5_spiny_idx = np.array(list(layer5_spiny))
layer5_spiny_idx.sort()
layer5_spiny_df = cell_feat_df.loc[layer5_spiny_idx]

def species():
    # if spex == 'Homo Sapiens':
    #     human_cells = cell_feat_df[cell_feat_df["donor__species"].values == spex].index
    # if spex == 'Mus musculus':
    #     mice_cells = cell_feat_df[cell_feat_df["donor__species"].values == spex].index
    human_cells = cell_feat_df[cell_feat_df["donor__species"].values == 'Homo Sapiens'].index
    mice_cells = cell_feat_df[cell_feat_df["donor__species"].values == 'Mus musculus'].index
    return human_cells, mice_cells

####################################################################################
# Changing variables

human_cells, mice_cells = species()
spex = 'Homo Sapiens' # 'Mus musculus'

layer5_spiny_human = set(cells_in_layer)&set(spiny_cells)& set(human_cells)
layer5_spiny_human_idx = np.array(list(layer5_spiny_human))
layer5_spiny_human_idx.sort()
layer5_spiny_human_df = cell_feat_df.loc[layer5_spiny_human_idx]

def layer_type_species(layer, spex, type):
    cells_in_layer = which_layer(layer)
    if spex == 'Homo Sapiens':
        spex_cells = human_cells 
    if spex == 'Mus musculus':
        spex_cells = mice_cells 
    if type == 'spiny':
        type_cells = spiny_cells
    if type == 'aspiny':
        type_cells = aspiny_cells
    layer_type_spex = set(cells_in_layer)&set(type_cells)&set(spex_cells)
    layer_type_spex_idx = np.array(list(layer_type_spex))
    layer_type_spex_idx.sort()
    layer_type_spex_idx_df = cell_feat_df.loc[layer_type_spex_idx]
    return layer_type_spex_idx_df

layer_type_spex_idx_df = layer_type_species(layer, spex, type)
    # layer_spiny_human = set(cells_in_layer)&set(spiny_cells)& set(human_cells)
    # layer_aspiny_human = set(cells_in_layer)&set(aspiny_cells)& set(human_cells)
    # layer_spiny_mice = set(cells_in_layer)&set(spiny_cells)& set(mice_cells)
    # layer_aspiny_mice = set(cells_in_layer)&set(aspiny_cells)& set(mice_cells)
