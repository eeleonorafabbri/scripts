# Cleanup
from IPython import get_ipython
get_ipython().magic('reset -sf') 

# Script for the new conda environment elphy_env
# I just dowloaded python=3.10 and NEURON with: pip3 install neuron
# Then I had to install pynwb with conda and allensdk with pip
import allensdk
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.api.queries.biophysical_api import \
    BiophysicalApi
from allensdk.api.queries.glif_api import GlifApi
from allensdk.model.biophys_sim.config import Config
import allensdk.core.json_utilities as json_utilities

from allensdk.model.biophysical import utils
from allensdk.model.biophysical.utils import Utils
from allensdk.model.biophysical.utils import AllActiveUtils
from allensdk.model.biophysical import runner
from allensdk.model.biophysical.runner import load_description

import Organised_Script

import pynwb
from pynwb import NWBHDF5IO
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import pdb

# #ATTENTION i need to set this environment variable BEFORE running this script, otherwise it doesn't work
# import os
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  #I nedd it to make the simulation loop work (otherwise the h5f files are unable to lock, whatever it means...)


#######################################################################################################################################################

# FUNCTIONS


def get_sweep_data(nwb_file, sweep_number, time_scale=1e3, voltage_scale=1e3, stim_scale=1e12):
    """
    Extract data and stim characteristics for a specific DC sweep from nwb file
    Parameters
    ----------
    nwb_file : string
        File name of a pre-existing NWB file.
    sweep_number : integer
        
    time_scale : float
        Convert to ms scale
    voltage_scale : float
        Convert to mV scale
    stim_scale : float
        Convert to pA scale

    Returns
    -------
    t : numpy array
        Sampled time points in ms
    v : numpy array
        Recorded voltage at the sampled time points in mV
    stim_start_time : float
        Stimulus start time in ms
    stim_end_time : float
        Stimulus end time in ms
    """
    nwb = NwbDataSet(nwb_file)
    sweep = nwb.get_sweep(sweep_number)
    stim = sweep['stimulus'] * stim_scale  # in pA
    stim_diff = np.diff(stim)
    stim_start = np.where(stim_diff != 0)[0][-2]
    stim_end = np.where(stim_diff != 0)[0][-1]
    
    # read v and t as numpy arrays
    v = sweep['response'] * voltage_scale  # in mV
    dt = time_scale / sweep['sampling_rate']  # in ms
    num_samples = len(v)
    t = np.arange(num_samples) * dt
    stim_start_time = t[stim_start]
    stim_end_time = t[stim_end]
    return t, v, stim_start_time, stim_end_time


# # I want to create a function that takes as input the specimen_id and if it's perisomatic or all_active, and returns a dataframe with the columns: 'section','name,'mechanism'
# def specimen_id_to_df(specimen_id, cell_type):
#     '''
#     This function takes as input the specimen_id and the cell_type (perisomatic or all_active), and returns a dataframe with the columns: 'section','name,'mechanism'
#     '''

#     specimen_id = str(specimen_id)
#     if cell_type == 'perisomatic':
#         base_dir_path = Path('/opt3/Eleonora/data/First_try_download/perisomatic')
#     elif cell_type == 'all_active':
#         base_dir_path = Path('/opt3/Eleonora/data/First_try_download/all_active')
#     else:
#         print('Error: cell_type must be perisomatic or all_active')
#     fit_json_path = base_dir_path / specimen_id / (specimen_id + '_fit.json')
#     with open(fit_json_path, 'r') as file:
#         fit_json_data = json.load(file)
#     section_list = []
#     mechanism_list = []
#     value_list = []
#     for item in fit_json_data['genome']:
#         section_list.append(item['section'])
#         mechanism_list.append(item['mechanism'])
#         value_list.append(item['value'])
#     # Create a pandas DataFrame
#     fit_df = pd.DataFrame({
#         'section': section_list,
#         'mechanism': mechanism_list,
#         'value': value_list
#     })
#     return fit_df


def get_fit_df(specimen_id, cell_type):
    '''
    This function takes as input the specimen_id and the cell_type (perisomatic or all_active), and returns a dataframe with the
    columns: 'section','name,'mechanism' (taken from its fit.json file)
    '''

    specimen_id_str = str(specimen_id)
    if cell_type == 'perisomatic':
        base_dir_path = Path('/opt3/Eleonora/data/First_try_download/perisomatic')
    elif cell_type == 'all_active':
        base_dir_path = Path('/opt3/Eleonora/data/First_try_download/all_active')
    else:
        print('Error: cell_type must be perisomatic or all_active')
    file_in_cartella = os.listdir(base_dir_path / specimen_id_str)
    for file in file_in_cartella:
        if file.endswith(".json") and "fit" in file:
            fit_json_path = base_dir_path / specimen_id_str / file
   # fit_json_path = base_dir_path / specimen_id_str / (specimen_id_str + '_fit.json')
            with open(fit_json_path, 'r') as file:
                fit_json_data = json.load(file)
            section_list = []
            name_list = []
            mechanism_list = []
            for item in fit_json_data['genome']:
                section_list.append(item['section'])
                name_list.append(item['name'])
                mechanism_list.append(item['mechanism'])
            # Create a pandas DataFrame
            fit_df = pd.DataFrame({
                'section': section_list,
                'name': name_list,
                'mechanism': mechanism_list
            })
            return fit_df


def get_values_from_fit(specimen_id, cell_type, section, name, mechanism):
    '''
    This function takes as input the specimen_id, the cell_type (perisomatic or all_active), the section, the name and the mechanism
    and returns the value of the parameter (taken from its fit.json file)
    Options:
     - section: 'somatic', 'basal', 'apical'
     - name: for all_active: 'g_pas', 'e_pas', 'cm', 'Ra', 'gbar_NaV', 'gbar_K_T', 'gbar_Kv2like',
             'gbar_Kv3_1', 'gbar_SK', 'gbar_Ca_HVA', 'gbar_Ca_LVA', 'gamma_CaDynamics',
                'decay_CaDynamics', 'gbar_Ih', 'gbar_Im_v2', 'gbar_Kd';
            for perisomatic: 'gbar_Im', 'gbar_Ih', 'gbar_NaTs', 'gbar_Nap', 'gbar_K_P',
             'gbar_K_T', 'gbar_SK', 'gbar_Kv3_1', 'gbar_Ca_HVA', 'gbar_Ca_LVA', 'gamma_CaDynamics',
                'decay_CaDynamics', 'g_pas', 'gbar_NaV', 'gbar_Kd', 'gbar_Kv2like', 'gbar_Im_v2';
     - mechanism: for all_active: '', 'NaV', 'K_T', 'Kd', 'Kv2like', 'Kv3_1', 'SK', 'Ca_HVA', 'Ca_LVA', 
                    'CaDynamics', 'Ih', 'Im_v2'
                for perisomatic: '', 'Im', 'Ih', 'NaTs', 'Nap', 'K_P', 'K_T', 'SK', 'Kv3_1', 'Ca_HVA', 'Ca_LVA', 
                    'CaDynamics', 'NaV', 'Kd', 'Kv2like', 'Im_v2'


    specimen_id_str = str(specimen_id)
    if cell_type == 'perisomatic':
        base_dir_path = Path('/opt3/Eleonora/data/First_try_download/perisomatic')
    elif cell_type == 'all_active':
        base_dir_path = Path('/opt3/Eleonora/data/First_try_download/all_active')
    else:
        print('Error: cell_type must be perisomatic or all_active')
    file_in_cartella = os.listdir(base_dir_path / specimen_id_str)
    for file in file_in_cartella:
        if file.endswith(".json") and "fit" in file:
            fit_json_path = base_dir_path / specimen_id_str / file
            with open(fit_json_path, 'r') as file:
                fit_json_data = json.load(file)
            for item in fit_json_data['genome']:
                if item['section'] == section and item['name'] == name and item['mechanism'] == mechanism:
                    value = item['value']
                    return value

#######################################################################################################################################################

# VARIABLE PARAMETERS

ctc = CellTypesCache(manifest_file="cell_types/manifest.json")

specimen_id = 323475862 

base_dir = '/opt3/Eleonora/scripts/messy_things/'

sweep_num = 4 # It's just an example

mV = 1.0e-3

ms = 1.0e-3




#x = Path.cwd()

# # a list of cell metadata for cells with reconstructions, download if necessary
# cells = ctc.get_cells(require_reconstruction=True)

# # open the electrophysiology data of one cell, download if necessary
# data_set = ctc.get_ephys_data(cells[0]["id"])

# # read the reconstruction, download if necessary
# reconstruction = ctc.get_reconstruction(cells[0]["id"])

data_set = ctc.get_ephys_data(specimen_id)
sweeps = ctc.get_ephys_sweeps(specimen_id)
sweep_numbers = defaultdict(list)
for sweep in sweeps:
    sweep_numbers[sweep["stimulus_name"]].append(sweep["sweep_number"])

# calculate features
cell_features = extract_cell_features(
    data_set,
    sweep_numbers["Ramp"],
    sweep_numbers["Short Square"],
    sweep_numbers["Long Square"],
)
# # if you ran the examples above, you will have a NWB file here
# file_name = "cell_types/specimen_485909730/ephys.nwb"
# data_set = NwbDataSet(file_name)

# sweep_numbers = data_set.get_sweep_numbers()
# sweep_number = sweep_numbers[0]
# sweep_data = data_set.get_sweep(sweep_number)

# # spike times are in seconds relative to the start of the sweep
# spike_times = data_set.get_spike_times(sweep_number)

# # stimulus is a numpy array in amps
# stimulus = sweep_data["stimulus"]

# # response is a numpy array in volts
# reponse = sweep_data["response"]

# # sampling rate is in Hz
# sampling_rate = sweep_data["sampling_rate"]

# # start/stop indices that exclude the experimental test pulse (if applicable)
# index_range = sweep_data["index_range"]


# I obtained the json files from 
# http://api.brain-map.org/api/v2/data/query.json?q=model::NeuronalModel,rma::include,neuronal_model_template[name$eq%27Biophysical%20-%20all%20active%27],rma::options[num_rows$eqall]
# and http://api.brain-map.org/api/v2/data/query.json?q=model::NeuronalModel,rma::include,neuronal_model_template[name$eq%27Biophysical%20-%20perisomatic%27],rma::options[num_rows$eqall]
# and http://api.brain-map.org/api/v2/data/query.json?criteria=model::ApiCellTypesSpecimenDetail,rma::criteria,[m__glif$gt0] for the GLIF cells
# I took them from the Allen Institute Git repository (model-biophysical-passive_fitting-biophysical_archiver)
with open('/opt3/Eleonora/data/query_perisomatic.json', 'r') as file:
    perisomatic_data = json.load(file)
with open('/opt3/Eleonora/data/query_all_active.json', 'r') as file:
    all_active_data = json.load(file)
with open('/opt3/Eleonora/data/query_glif.json', 'r') as file:
    glif_data = json.load(file)


#this list contains the neuronal_models_ids of the cells that have biophysical perisoamtic models
# perisomatic_id_list = [item["id"] for item in perisomatic_data["msg"]] 
msg_list = perisomatic_data["msg"]
msg_peri_df = pd.json_normalize(msg_list)
perisomatic_id_list = msg_peri_df.id



#this list contains the neuronal_models_ids of the cells that have biophysical all_active models
# all_active_id_list = [item["id"] for item in all_active_data["msg"]]
msg_list = all_active_data["msg"]
msg_all_act_df = pd.json_normalize(msg_list)
all_active_id_list = msg_all_act_df.id


# #neur_id = 497233139
# neur_id = 478513459
# bp = BiophysicalApi()
# bp.cache_stimulus = True # change to False to not download the large stimulus NWB file
# neuronal_model_id = neur_id    # get this from the web site as above
# bp.cache_data(neuronal_model_id, working_directory='messy_things')

os.chdir(base_dir)
os.system('nrnivmodl modfiles/')
manifest_file = 'manifest.json'
manifest_dict = json.load(open(manifest_file))
# base_dir = Path('/opt3/Eleonora/scripts/messy_things')
# manifest_file = base_dir / 'manifest.json'
# manifest_dict = json.load(open(manifest_file))



if 'sweeps_by_type' not in manifest_dict['runs'][0]:
    manifest_dict['runs'][0]['sweeps_by_type'] = {'Long Square': sweep_numbers['Long Square'], 'Short Square': sweep_numbers['Short Square']}
json.dump(manifest_dict,open(manifest_file,'w'),indent=2)

# sweep_num = 43 # It's one of the sweeps of the Long Square

schema_legacy = dict(manifest_file=manifest_file)
# runner.run(schema_legacy,procs=1,sweeps=[sweep_num])


# After running this script you should exit from ipython and run the following command in the terminal:
# cd messy_things/ (or the place where you put the nwb, modfiles and manifest.json)
# nrnivmodl modfiles/
# python -m allensdk.model.biophysical.runner manifest.json


my_args_dict={'manifest_file' : '/opt3/Eleonora/scripts/messy_things/manifest.json', 'axon_type' : 'truncated'}
# my_args_dict={'manifest_file' : str(manifest_file), 'axon_type' : 'truncated'}
description = load_description(my_args_dict)
# allensdk.model.biophysical.runner.load_description(my_args_dict)
my_utils = utils.create_utils(description)
h = my_utils.h
manifest = description.manifest
# pdb.set_trace()
morphology_path = manifest.get_path('MORPHOLOGY')
# my_utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
my_utils.generate_morphology(morphology_path)
# my_utils.load_cell_parameters()
stimulus_path = manifest.get_path('stimulus_path')
nwb_out_path = manifest.get_path("output_path")  #I changed this line from the original simulation loop from the Allen example
output = NwbDataSet(nwb_out_path)
run_params = description.data['runs'][0]
sweeps = run_params['sweeps']
junction_potential = description.data['fitting'][0]['junction_potential']
# for sweep in sweeps:
#     # try:
#     #     my_utils.setup_iclamp(stimulus_path, sweep=sweep)
#     # except:
#     #     pdb.set_trace()

#     # configure stimulus
#     my_utils.setup_iclamp(stimulus_path, sweep=sweep)

#     # configure recording
#     vec = my_utils.record_values()

#     h.finitialize()
#     h.run()

#     # write to an NWB File
#     output_data = (np.array(vec['v']) - junction_potential) * mV
#     # try:
#     #     output.set_sweep(sweep, None, output_data)
#     # except:
#     #     pdb.set_trace()
#     output.set_sweep(sweep, None, output_data)



t, v, stim_start, stim_end = get_sweep_data(stimulus_path, sweep_num)
plt.plot(t, v)
os.chdir('/opt3/Eleonora/scripts')

# It's possible to use the functions in AllenSDK.doc_template.examples_root.examples.simple.utils, that are much more quicker because it doesn't
# need to analize each sweep in a for loop
from AllenSDK.doc_template.examples_root.examples.simple.utils import Utils

other_utils = Utils(description)
hh = other_utils.h
manifest = description.manifest
other_utils.generate_morphology()
other_utils.setup_iclamp()
vec = other_utils.record_values()
hh.dt = 0.025
hh.tstop = 20
hh.finitialize()
hh.run()
output_data2 = np.array(vec['v']) * mV
output_times2 = np.array(vec['t']) * ms
output2 = np.column_stack((output_times2, output_data2))
# write to a dat File
v_out_path = manifest.get_path("output_path")
with open (v_out_path, "w") as f:
    np.savetxt(f, output2)



# # I want to create a dataframe out of my fit.json, with columns: 'section','name,'mechanism' 
# fit_json_path = '/opt3/Eleonora/data/First_try_download/perisomatic/471410185/471410185_fit.json'
# with open(fit_json_path, 'r') as file:
#     fit_json_data = json.load(file)
# section_list = []
# mechanism_list = []
# value_list = []
# for item in fit_json_data['genome']:
#     section_list.append(item['section'])
#     mechanism_list.append(item['mechanism'])
#     value_list.append(item['value'])

# # Create a pandas DataFrame
# fit_df = pd.DataFrame({
#     'section': section_list,
#     'mechanism': mechanism_list,
#     'value': value_list
# })



# #cell_feat_orient_df = pd.read_csv("/opt3/Eleonora/data/cell_feat_orientation_data.csv")

# common_id = set(msg_peri_df.specimen_id)&set(cell_feat_orient_df.specimen__id)
# # From the idx mice_cells I had from Organised_Script I want to take the specimen_id from cell_feat_orient_df
# name_values = []
# for idx in human_cells:
#     value = cell_feat_orient_df.loc[idx, 'specimen__id']
#     name_values.append(value)





