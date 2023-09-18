# IMPORTS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import csv
#from sklearn.decomposition import PCA  # I have commented this line because it was giving me an error importing Organised_Script, but maybe it's necessary for this script
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve, newton, bisect

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.core.cell_types_cache import ReporterStatus as RS
from allensdk.api.queries.biophysical_api import BiophysicalApi
from allensdk.api.queries.cell_types_api import CellTypesApi

# from examples.rotation_cell_morph_example import morph_func as mf
# import Organised_Script


from pathlib import Path
import math
import pprint
import pdb


class Analysis:
    def __init__(self, output_dir, ctc):
        self.output_dir = Path("/opt3/Eleonora/data/reconstruction")
        self.ctc = CellTypesCache(manifest_file=self.output_dir / "manifest.json")

    def reconstruct(self, id_cell):
        morphology = self.ctc.get_reconstruction(id_cell)
        return morphology

    def proper_rotation(self, slice_angle, upright_angle, x1, y1, z1, shrink_factor):
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

    def get_cell_morphXYZ(self, cell_id):
        morph = self.ctc.get_reconstruction(cell_id)
        x = []
        y = []
        z = []
        for n in morph.compartment_list:
            # print(n['type']) #type=1, soma; type=2, axon; type=3, dendrite; type=4,apical dendrite
            if n["type"] == 4 or n["type"] == 3 or n["type"] == 1:
                x.append(n["x"] - morph.soma["x"])
                y.append(n["y"] - morph.soma["y"])
                z.append(n["z"] - morph.soma["z"])

        morph_data = np.array(np.column_stack((x, y, z)))
        morph_soma = [morph.soma["x"], morph.soma["y"], morph.soma["z"]]

        return (morph_data, morph_soma)

    def cal_rotation_angle(self, morph_data):
        pca = PCA(n_components=2)
        pca.fit(morph_data)
        proj = morph_data.dot(
            pca.components_[0]
        )  # the projection of morphology on the direction of first pca
        # v1 = -1*pca.components_[0]  # the first principal component, when apical dendrite goes down
        # v1 = 1*pca.components_[0]  # the first principal component
        v1 = np.sign(proj.mean()) * pca.components_[0]
        # The goal is to rotate v1 to parallel to y axis
        x1 = v1[0]
        y1 = v1[1]
        z1 = v1[2]
        # First rotate in the anticlockwise direction around z axis untill x=0
        v2 = [0, math.sqrt(y1 * y1 + x1 * x1), z1]
        dv = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
        anglez = 2 * math.asin(
            math.sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]) * 0.5 / v2[1]
        )
        if x1 < 0:  # when x1 in the negative side, change the sign
            anglez = -anglez
        # Second rotate in the anticlockwise direction round x axis untill z = 0
        v3 = [0, math.sqrt(x1 * x1 + y1 * y1 + z1 * z1), 0]
        dv2 = [v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2]]
        anglex = -2 * math.asin(
            math.sqrt(dv2[0] * dv2[0] + dv2[1] * dv2[1] + dv2[2] * dv2[2]) * 0.5 / v3[1]
        )
        if z1 < 0:  # when z1 in the negative side, change the sign
            anglex = -anglex
        theta = [anglex, 0, anglez]
        R = self.eulerAnglesToRotationMatrix(theta)
        v3_hat = R.dot(v1)

        return (v1, theta)

    def eulerAnglesToRotationMatrix(self, theta):
        R_x = np.array(
            [
                [1, 0, 0],
                [0, math.cos(theta[0]), -math.sin(theta[0])],
                [0, math.sin(theta[0]), math.cos(theta[0])],
            ]
        )

        R_y = np.array(
            [
                [math.cos(theta[1]), 0, math.sin(theta[1])],
                [0, 1, 0],
                [-math.sin(theta[1]), 0, math.cos(theta[1])],
            ]
        )

        R_z = np.array(
            [
                [math.cos(theta[2]), -math.sin(theta[2]), 0],
                [math.sin(theta[2]), math.cos(theta[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(R_x, np.dot(R_y, R_z))

        return R

    def cell_morphology_rot(self, cell_id, x_soma, y_soma, z_soma, theta):
        theta_z = theta[2]
        theta_y = theta[1]
        theta_x = theta[0]
        morph = self.ctc.get_reconstruction(cell_id)
        # First applying a rotation angle around z axis
        tr_rot_z = [
            math.cos(theta_z),
            -math.sin(theta_z),
            0,
            math.sin(theta_z),
            math.cos(theta_z),
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ]
        # Second applying a rotation angle around y axis
        tr_rot_y = [
            math.cos(theta_y),
            0,
            math.sin(theta_y),
            0,
            1,
            0,
            -math.sin(theta_y),
            0,
            math.cos(theta_y),
            0,
            0,
            0,
        ]
        # Third applying a rotation angle around x axis
        tr_rot_x = [
            1,
            0,
            0,
            0,
            math.cos(theta_x),
            -math.sin(theta_x),
            0,
            math.sin(theta_x),
            math.cos(theta_x),
            0,
            0,
            0,
        ]

        morph.apply_affine(tr_rot_z)
        morph.apply_affine(tr_rot_y)
        morph.apply_affine(tr_rot_x)
        # translate the soma location
        tr_soma = [
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            -morph.soma["x"] + x_soma,
            -morph.soma["y"] + y_soma,
            -morph.soma["z"] + z_soma,
        ]
        morph.apply_affine(tr_soma)
        return morph

    def plot_cell_morph_xyzy(self, axes, morph):
        soma_col = [134.0 / 255.0, 134.0 / 255.0, 148.0 / 255.0]
        axon_col = [93.0 / 255.0, 127.0 / 255.0, 177.0 / 255.0]
        dend_col = [153.0 / 255.0, 40.0 / 255.0, 39.0 / 255.0]
        apical_dend_col = [227.0 / 255.0, 126.0 / 255.0, 39.0 / 255.0]
        ap = 1

        for n in morph.compartment_list:
            for c in morph.children_of(n):
                if n["type"] == 2:
                    axes[0].plot(
                        [n["x"], c["x"]], [n["y"], c["y"]], color=axon_col, alpha=ap
                    )
                    axes[1].plot(
                        [n["z"], c["z"]], [n["y"], c["y"]], color=axon_col, alpha=ap
                    )
                if n["type"] == 3:
                    axes[0].plot(
                        [n["x"], c["x"]], [n["y"], c["y"]], color=dend_col, alpha=ap
                    )
                    axes[1].plot(
                        [n["z"], c["z"]], [n["y"], c["y"]], color=dend_col, alpha=ap
                    )
                if n["type"] == 4:
                    axes[0].plot(
                        [n["x"], c["x"]],
                        [n["y"], c["y"]],
                        color=apical_dend_col,
                        alpha=ap,
                    )
                    axes[1].plot(
                        [n["z"], c["z"]],
                        [n["y"], c["y"]],
                        color=apical_dend_col,
                        alpha=ap,
                    )
                if n["type"] == 1:  # soma
                    axes[0].scatter(
                        n["x"], n["y"], s=math.pi * (n["radius"] ** 2), color=soma_col
                    )
                    axes[1].scatter(
                        n["z"], n["y"], s=math.pi * (n["radius"] ** 2), color=soma_col
                    )

        axes[0].set_ylabel("y")
        axes[0].set_xlabel("x")
        axes[1].set_xlabel("z")
        self.simpleaxis(axes[0])
        self.simpleaxis(axes[1])

    def simpleaxis(self, ax):
        # Hide the right and top spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    def get_rotation_theta(self, cell_id):
        # get the morphology data (from apical dendrite, dendrite, and soma) used for PCA
        [morph_data, morph_soma] = self.get_cell_morphXYZ(cell_id)
        [v, theta] = self.cal_rotation_angle(morph_data)
        return theta


class Viz(Analysis):
    def __init__(self):
        self.axon_color = "blue"
        self.bas_dendrite_color = "red"
        self.api_dendrite_color = "orange"
        self.output_dir = Path("/opt3/Eleonora/data/reconstruction")
        self.ctc = CellTypesCache(manifest_file=self.output_dir / "manifest.json")
        # self.ct = CellTypesApi()

    def show_neuron_2D(self, cell_id):
        """
        This function shows the neuron shape in 2D (x and y) thru the coordinates given  in the morphology file.
        """

        # Get the morphology file using the cell_id of the neuron
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)

        fig, ax = plt.subplots(1, 1)

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            ax.scatter(df["x"], df["y"], color=color)
        ax.invert_yaxis()
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])

    def show_orient_neuron_2D(self, cell_id, cell_feat_orient_new_df):
        """
        This function shows the neuron shape in 2D (x and y) thru the coordinates that have been properly rotated using the slice_angle and upright_angle,
        given in the cell_feat_orient_df dataframe
        """
        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)
        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        fig, ax = plt.subplots(1, 1)

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
            )
            ax.scatter(x_coord, y_coord, color=color)
        ax.invert_yaxis()
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])

    def show_orient_neuron_3D(self, cell_id, cell_feat_orient_new_df):
        """
        This function shows the neuron shape in 3D (x, y and z) thru the coordinates properly rotated
        """
        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)
        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
            )
        ax.scatter(x_coord, z_coord, y_coord, color=color)
        ax.invert_yaxis()  # Maybe it is not necessary
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])

    def plot_difference(self, cell_id, cell_feat_orient_new_df):
        """
        This function creates 3 different plots, the first two from the pca coordinates and the third one from the coordinates
        that have been properly rotated using the slice_angle and upright_angle,
        """

        # FIRST PLOT: PCA COORDINATES

        theta = list(np.around(self.get_rotation_theta(cell_id), decimals=6))

        # get the morphology data (from apical dendrite, dendrite, and soma) used for PCA
        [morph_data, morph_soma] = self.get_cell_morphXYZ(cell_id)

        # get the first principal vector and the rotation angle
        [v, theta] = self.cal_rotation_angle(morph_data)

        # Based on the rotation angle to calculate the rotation matrix R
        R = self.eulerAnglesToRotationMatrix(theta)  # rotation matrix

        # the first principal component before and after rotated
        v = v * 400
        v_rot = R.dot(v)
        # The morphology locations used for PCA after rotations
        X_rot = np.array(morph_data)  # The rotated position of new x,y,z
        for i in range(0, len(X_rot)):
            X_rot[i, :] = R.dot(morph_data[i, :])

        # The location of soma, defined by the user
        x_soma = 0
        y_soma = 0
        z_soma = 0
        # The original morphology before rotations
        theta0 = [0, 0, 0]
        morph0 = self.cell_morphology_rot(cell_id, x_soma, y_soma, z_soma, theta0)

        # Plot the morphology in xy and zy axis
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        self.plot_cell_morph_xyzy(axes, morph0)
        # plot the principal vectors on top of the morphology
        axes[0].plot([x_soma, v[0]], [y_soma, v[1]], color="c")
        axes[1].plot([x_soma, v[2]], [y_soma, v[1]], color="c")
        axes[0].scatter(v[0], v[1], color="blue")
        axes[1].scatter(v[2], v[1], color="blue")

        # The morphology after rotations
        morph_rot = self.cell_morphology_rot(cell_id, x_soma, y_soma, z_soma, theta)

        # Plot the morphology in xy and zy axis
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        self.plot_cell_morph_xyzy(axes, morph_rot)
        # plot the principal vectors on top of the morphology
        axes[0].plot([x_soma, v_rot[0]], [y_soma, v_rot[1]], color="c")
        axes[1].plot([x_soma, v_rot[2]], [y_soma, v_rot[1]], color="c")
        axes[0].scatter(v_rot[0], v_rot[1], color="blue")
        axes[1].scatter(v_rot[2], v_rot[1], color="blue")

        # SECOND PLOT: ROTATED COORDINATES

        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)

        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        fig, ax = plt.subplots(1, 1)

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            # shrink_z_coord = df['z'] *shrink_factor
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
            )
            ax.scatter(x_coord, y_coord, color=color)
        # ax.invert_xaxis()
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])

    def y_coord_difference(self, specimens, cell_feat_orient_new_df):
        """
        This function plots the histogram of the difference between the maximum and minimum y
        coordinate (rotated) of the neurons, compared to the pca coordinates
        """
        minmax_difference = []
        for cell_id in specimens:
            cell_idx = cell_feat_orient_new_df[
                cell_feat_orient_new_df["specimen_id"] == cell_id
            ].index
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ].values

            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            minmax_difference.append(max(y_coord) - min(y_coord))

        fig, ax = plt.subplots(1, 1)
        plt.hist(
            minmax_difference,
            bins=50,
            orientation="horizontal",
            label="our_rotated_coord",
            color="tab:orange",
        )

        x_soma = 0
        y_soma = 0
        z_soma = 0
        pca_rotated_coord = []
        for cell_id in specimens:
            [morph_data, morph_soma] = self.get_cell_morphXYZ(cell_id)
            [v, theta] = self.cal_rotation_angle(morph_data)
            R = self.eulerAnglesToRotationMatrix(theta)  # rotation matrix
            v = v * 400
            v_rot = R.dot(v)
            X_rot = np.array(morph_data)  # The rotated position of new x,y,z
            for i in range(0, len(X_rot)):
                X_rot[i, :] = R.dot(morph_data[i, :])
            morph_rot = self.cell_morphology_rot(cell_id, x_soma, y_soma, z_soma, theta)
            morph_rot_df = pd.DataFrame(morph_rot.compartment_list)
            pca_rotated_coord.append(max(morph_rot_df["y"]) - min(morph_rot_df["y"]))
        plt.hist(
            pca_rotated_coord,
            bins=50,
            orientation="horizontal",
            label="pca_rotated_coord",
            color="tab:purple",
        )
        plt.legend()

    def soma_coord_and_pia_distance(self, specimens, cell_feat_orient_new_df):
        """
        This function plots the histogram of the soma distance from the pia and the soma y coordinate (after rotation)
        """
        soma_y_coord = []
        soma_distance = []
        for cell_id in specimens:
            cell_idx = cell_feat_orient_new_df[
                cell_feat_orient_new_df["specimen_id"] == cell_id
            ].index
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ].values

            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            soma_y_coord.append(y_coord[0])
            soma_dist = cell_feat_orient_new_df.loc[
                cell_idx, "soma_distance_from_pia"
            ].to_list()
            soma_distance.append(soma_dist)
        soma_distance = np.concatenate(soma_distance).tolist()
        soma_y_coord = [-x for x in soma_y_coord]  # because i have inverted the y axis

        plt.hist(
            soma_distance,
            bins=50,
            orientation="horizontal",
            label="soma_distance_from_pia",
            color="yellowgreen",
        )
        plt.hist(
            soma_y_coord,
            bins=50,
            orientation="horizontal",
            label="soma_y_coord",
            color="olivedrab",
        )
        plt.legend()
        return soma_distance, soma_y_coord

    def plot_morphology_from_pia(self, cell_id, cell_feat_orient_new_df):
        """
        This is the plot of a single neuron with proper rotation and with the origin in the pia
        """

        fig, ax = plt.subplots(1, 1)

        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)

        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        x_soma, y_soma, z_soma = self.proper_rotation(
            slice_angle,
            upright_angle,
            morph_df.loc[0, "x"],
            morph_df.loc[0, "y"],
            morph_df.loc[0, "z"],
            shrink_factor,
        )
        soma_distance_from_pia = cell_feat_orient_new_df.loc[
            cell_idx, "soma_distance_from_pia"
        ].values
        x_pia = x_soma
        y_pia = y_soma + soma_distance_from_pia

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
            )
            x_coord = [val - x_pia for val in x_coord]
            y_coord = [val - y_pia for val in y_coord]
            # x_coord = -x_coord
            # y_coord = -y_coord
            ax.scatter(x_coord, y_coord, color=color)
            # x_node, y_node, z_node = proper_rotation(
            #     slice_angle,
            #     upright_angle,
            #     morph_df.loc[idx, "x"],
            #     morph_df.loc[idx, "y"],
            #     morph_df.loc[idx, "z"],
            #     shrink_factor,
            # )
        # ax.invert_yaxis()
        # ax.invert_xaxis()

        plt.ylabel("y")
        plt.xlabel("x")
        # plt.savefig("bad_cell_2.png")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])
        return ax

    def cortical_depth_hist(self, cell_id, cell_feat_orient_new_df):
        """
        This function plots the histogram of the cortical depth of the neurites
        """
        fig, ax = plt.subplots(1, 1)
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)
        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        x_soma, y_soma, z_soma = self.proper_rotation(
            slice_angle,
            upright_angle,
            morph_df.loc[0, "x"],
            morph_df.loc[0, "y"],
            morph_df.loc[0, "z"],
            shrink_factor,
        )
        soma_distance_from_pia = cell_feat_orient_new_df.loc[
            cell_idx, "soma_distance_from_pia"
        ].values
        x_pia = x_soma
        y_pia = y_soma + soma_distance_from_pia
        x_coord, y_coord, z_coord = self.proper_rotation(
            slice_angle,
            upright_angle,
            morph_df["x"],
            morph_df["y"],
            morph_df["z"],
            shrink_factor,
        )
        x_coord = [val - x_pia for val in x_coord]
        y_coord = [val - y_pia for val in y_coord]
        ax.hist(
            np.array(y_coord).flatten(),
            orientation="horizontal",
            label="depth_from_pia",
            color="lightcoral",
        )
        plt.ylabel("depth")
        plt.xlabel("number")

    def _scatter_soma_position(self, cells_idx, cell_feat_orient_new_df):
        """
        This function plots the soma rotated position of the cells in the cells_idx list
        (example of cells_idx = mice_spiny_orient_cells_idx, so a list of cells index),
        using the layer type as color
        """
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

        for cell_idx in cells_idx:
            l_type = cell_feat_orient_new_df.loc[cell_idx, "structure__layer"]
            color = col_dict[l_type]
            cell_id = cell_feat_orient_new_df.loc[cell_idx, "specimen_id"]
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ]

            # x_soma, y_soma, z_soma = self.proper_rotation(
            #     slice_angle,
            #     upright_angle,
            #     morph_df.loc[0, "x"],
            #     morph_df.loc[0, "y"],
            #     morph_df.loc[0, "z"],
            #     shrink_factor,
            # )
            # ax.scatter(x_soma, y_soma, color=color, label=color)

            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            x_soma = x_coord[0]
            y_soma = y_coord[0]
            soma_distance_from_pia = cell_feat_orient_new_df.loc[
                cell_idx, "soma_distance_from_pia"
            ].values

            ax.scatter(x_coord[0], y_coord[0], color=color, label=color)

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

    def _soma_distance_hist(self, cell_feat_orient_new_df):
        """
        This function plots the histogram of the soma distance from the pia for all the cells in
        cell_feat_orient_new_df
        """
        fig, ax = plt.subplots(1, 1)
        plt.hist(
            cell_feat_orient_new_df.soma_distance_from_pia,
            orientation="horizontal",
            color="m",
        )
        ax.invert_yaxis()
        plt.ylabel("depth")
        plt.xlabel("number")

    def _soma_distance_hist_layer(self, cells_idx, cell_feat_orient_new_df):
        """
        This function plots the histogram of the soma distance from the pia for the cells in
        the cells_idx list (example of cells_idx = mice_spiny_orient_cells_idx, so a list
        of cells index), using the layer type as color
        """
        depth = []
        layer = []
        layer_color_dict = {
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

        # Loop to get one layer at the time
        for cell_idx in cells_idx:
            l_type = cell_feat_orient_new_df.loc[cell_idx, "structure__layer"]
            layer.append(layer_color_dict[l_type])
            soma_distance_from_pia = cell_feat_orient_new_df.loc[
                cell_idx, "soma_distance_from_pia"
            ]
            depth.append(soma_distance_from_pia)

        depth_np = np.array(depth)
        layer_np = np.array(layer)

        fig, ax1 = plt.subplots(1, 1)

        # Loop to plot the histogram of the soma distance from the pia for each layer
        for l in layer_color_dict.values():
            # plt.hist(
            #     depth_np[layer_np == l],
            #     orientation="horizontal",
            #     color=l,
            #     label=l,
            #     alpha=0.5,
            # )
            if l == "r":
                data1 = depth_np[layer_np == l]
            if l == "#ff7f0e":
                data2 = depth_np[layer_np == l]
            if l == "y":
                data2_3 = depth_np[layer_np == l]
            if l == "g":
                data3 = depth_np[layer_np == l]
            if l == "c":
                data4 = depth_np[layer_np == l]
            if l == "b":
                data5 = depth_np[layer_np == l]
            if l == "#9467bd":
                data6 = depth_np[layer_np == l]
            if l == "#e377c2":
                data6a = depth_np[layer_np == l]
            if l == "#7f7f7f":
                data6b = depth_np[layer_np == l]
            sns.histplot(
                y=depth_np[layer_np == l],
                color=l,
                label=l,
                multiple="stack",
                kde=True,
            )

        fig, ax2 = plt.subplots(1, 1)
        for l in layer_color_dict.values():
            sns.histplot(
                y=depth_np[layer_np == l],
                color=l,
                label=l,
                element="poly",
            )

        # pdb.set_trace()

        red = mpatches.Patch(color="r", label="Layer 1")
        orange = mpatches.Patch(color="#ff7f0e", label="Layer 2")
        yellow = mpatches.Patch(color="y", label="Layer 2/3")
        green = mpatches.Patch(color="g", label="Layer 3")
        cian = mpatches.Patch(color="c", label="Layer 4")
        blue = mpatches.Patch(color="b", label="Layer 5")
        purple = mpatches.Patch(color="#9467bd", label="Layer 6")
        pink = mpatches.Patch(color="#e377c2", label="Layer 6a")
        grey = mpatches.Patch(color="#7f7f7f", label="Layer 6b")
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        plt.ylabel("soma_depth")
        plt.xlabel("x")
        plt.legend(handles=[red, orange, yellow, green, cian, blue, purple, pink, grey])

        return data1, data2, data2_3, data3, data4, data5, data6, data6a, data6b

    def _soma_y_coord_hist_layer(self, cells_idx, cell_feat_orient_new_df):
        """
        This function plots the histogram of the soma y rotated coordinate for the cells in
        the cells_idx list (example of cells_idx = mice_spiny_orient_cells_idx, so a list
        of cells index), using the layer type as color
        """
        soma_y_coord = []
        layer = []
        layer_color_dict = {
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
        # Loop to get one layer at the time
        for cell_idx in cells_idx:
            cell_id = cell_feat_orient_new_df.loc[cell_idx, "specimen_id"]
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ]
            l_type = cell_feat_orient_new_df.loc[cell_idx, "structure__layer"]
            layer.append(layer_color_dict[l_type])
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            y_soma = y_coord[0]
            soma_y_coord.append(y_soma)

        soma_y_coord_np = np.array(soma_y_coord)
        layer_np = np.array(layer)

        fig, ax = plt.subplots(1, 1)

        # Loop to plot the histogram of the soma distance from the pia for each layer
        for l in layer_color_dict.values():
            plt.hist(
                soma_y_coord_np[layer_np == l],
                orientation="horizontal",
                color=l,
                label=l,
                alpha=0.5,
            )

        red = mpatches.Patch(color="r", label="Layer 1")
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

    def soma_y_coord_and_distance_scatter(self, cells_idx, cell_feat_orient_new_df):
        fig, ax = plt.subplots(1, 1)
        depth = []
        soma_y_coord = []
        for cell_idx in cells_idx:
            soma_distance_from_pia = cell_feat_orient_new_df.loc[
                cell_idx, "soma_distance_from_pia"
            ]
            depth.append(soma_distance_from_pia)
            cell_id = cell_feat_orient_new_df.loc[cell_idx, "specimen_id"]
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ]
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            y_pia = y_coord[0] + soma_distance_from_pia
            y_coord = [val - y_pia for val in y_coord]
            soma_y_coord.append(y_coord[0])
        soma_y_coord = [-y for y in soma_y_coord]
        ax.scatter(depth, soma_y_coord)

    def layer_boundaries(self, data1, data2_3, data4, data5, data6a, data6b):
        """
        This function plots the histogram of layer distribution and the boundaries
        between the layers, for VISp mice cells.
        The data input are the soma distance from the pia for each layer, abained
        with the function _soma_distance_hist_layer.
        """
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=data1,
            kde=True,
            color="red",
            orientation="horizontal",
            label="Distribution 1",
        )
        sns.histplot(
            data=data2_3,
            kde=True,
            color="y",
            orientation="horizontal",
            label="Distribution 2_3",
        )
        sns.histplot(
            data=data4,
            kde=True,
            color="c",
            orientation="horizontal",
            label="Distribution 4",
        )
        sns.histplot(
            data=data5,
            kde=True,
            color="b",
            orientation="horizontal",
            label="Distribution 5",
        )
        sns.histplot(
            data=data6a,
            kde=True,
            color="pink",
            orientation="horizontal",
            label="Distribution 6a",
        )
        sns.histplot(
            data=data6b,
            kde=True,
            color="grey",
            orientation="horizontal",
            label="Distribution 6b",
        )

        kde1 = stats.gaussian_kde(data1)
        kde2_3 = stats.gaussian_kde(data2_3)
        kde4 = stats.gaussian_kde(data4)
        kde5 = stats.gaussian_kde(data5)
        kde6a = stats.gaussian_kde(data6a)
        kde6b = stats.gaussian_kde(data6b)

        def kde_diff123(x):
            return kde1.evaluate(x) - kde2_3.evaluate(x)

        def kde_diff234(x):
            return kde2_3.evaluate(x) - kde4.evaluate(x)

        def kde_diff45(x):
            return kde4.evaluate(x) - kde5.evaluate(x)

        def kde_diff56a(x):
            return kde5.evaluate(x) - kde6a.evaluate(x)

        def kde_diff6ab(x):
            return kde6a.evaluate(x) - kde6b.evaluate(x)

        def find_intersection(func, x0, x1):
            return bisect(func, x0, x1)

        # Find the intersection points with binary search
        intersection_point_123 = find_intersection(
            kde_diff123,
            min(data1.min(), data2_3.min()),
            max(data1.max(), data2_3.max()),
        )
        intersection_point_234 = find_intersection(
            kde_diff234,
            min(data2_3.min(), data4.min()),
            max(data2_3.max(), data4.max()),
        )
        intersection_point_45 = find_intersection(
            kde_diff45, min(data4.min(), data5.min()), max(data4.max(), data5.max())
        )
        intersection_point_56a = find_intersection(
            kde_diff56a, min(data5.min(), data6a.min()), max(data5.max(), data6a.max())
        )
        intersection_point_6ab = find_intersection(
            kde_diff6ab,
            min(data6a.min(), data6b.min()),
            max(data6a.max(), data6b.max()),
        )

        # Plot vertical lines at the intersection points to get the boundaries
        plt.axvline(
            x=intersection_point_123, color="k", linestyle="--", label="Boundary 1"
        )
        plt.axvline(
            x=intersection_point_234, color="k", linestyle="--", label="Boundary 2"
        )
        plt.axvline(
            x=intersection_point_45, color="k", linestyle="--", label="Boundary 3"
        )
        plt.axvline(
            x=intersection_point_56a, color="k", linestyle="--", label="Boundary 4"
        )
        plt.axvline(
            x=intersection_point_6ab, color="k", linestyle="--", label="Boundary 5"
        )

        plt.legend()
        return (
            intersection_point_123,
            intersection_point_234,
            intersection_point_45,
            intersection_point_56a,
            intersection_point_6ab,
        )

    def plot_in_layer(self, cell_id, cell_feat_orient_new_df, cells_idx):
        """
        This function plots the 2D morphology of a single neuron in the layer it belongs to
        """

        (
            data1,
            data2,
            data2_3,
            data3,
            data4,
            data5,
            data6,
            data6a,
            data6b,
        ) = self._soma_distance_hist_layer(cells_idx, cell_feat_orient_new_df)

        (
            intersection_point_123,
            intersection_point_234,
            intersection_point_45,
            intersection_point_56a,
            intersection_point_6ab,
        ) = self.layer_boundaries(data1, data2_3, data4, data5, data6a, data6b)

        ax = self.plot_morphology_from_pia(cell_id, cell_feat_orient_new_df)
        # I want to change the sign of every intersection point
        intersection_point_123 = -intersection_point_123
        intersection_point_234 = -intersection_point_234
        intersection_point_45 = -intersection_point_45
        intersection_point_56a = -intersection_point_56a
        intersection_point_6ab = -intersection_point_6ab

        ax.set_xlabel("Soma distance from pia")
        ax.set_ylabel("Number of cells")
        ax.axhline(0, color="k", linestyle="--")
        ax.axhline(intersection_point_123, color="k", linestyle="--")
        ax.axhline(intersection_point_234, color="k", linestyle="--")
        ax.axhline(intersection_point_45, color="k", linestyle="--")
        ax.axhline(intersection_point_56a, color="k", linestyle="--")
        ax.axhline(intersection_point_6ab, color="k", linestyle="--")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])
