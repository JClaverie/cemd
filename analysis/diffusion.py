#
# This file is part of the CEMD distribution
# Copyright (c) 2024-2026 Jérôme Claverie.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING
import scipy.stats as stats
import matplotlib.pyplot as plt


from .._utils import lattice2vectors

if TYPE_CHECKING:
    import MDAnalysis as mda


def _linear_fit(t: np.ndarray, 
                msd: np.ndarray, 
                sigma_msd: np.ndarray=None) -> tuple[float, float]:

    if sigma_msd is not None:
        w = 1 / sigma_msd

        fit, cov_matrix = np.polyfit(t, msd, 1, w=w, cov='unscaled')

    else:
        fit, cov_matrix = np.polyfit(t, msd, 1, cov=True)

    m, _ = fit

    # Extract standard errors from the scaled covariance matrix
    se_m_cov = np.sqrt(cov_matrix[0, 0])

    # Calculate the confidence intervals (e.g., 95% confidence)
    alpha = 0.05
    t_value = stats.t.ppf(1 - alpha / 2, df=len(t) - 2)
    ci_m = t_value * se_m_cov

    return m, ci_m


def msd(univ: mda.Universe, 
             atom_type: str, 
             dt: float=100.0, 
             nblocks: int=None, 
             corrlength: int=None, 
             gaplength: int=None, 
             csv_file: str=None) -> pd.DataFrame:
    """Calculate the diffusion coefficient in a bulk solution from the mean squared displacement (MSD) according to Einstein's equation:

    .. math:: D = \\frac{1}{6} \\frac{\\langle [\\mathbf{r}(t_0 + \\tau) - \\mathbf{r}(t_0)]^2 \\rangle}{\\tau} = \\frac{1}{6} \\frac{\\text{MSD}}{\\tau}

    The positions of the atom selection are stored at each time origin.

    Parameters
    ----------
        univ: MDAnalysis.Universe
            The input MDAnalysis Universe to analyse
        atom_type: str
            The atom type to compute the diffusion coefficient
        dt: float
            Timestep between each frame of the trajectory (fs)
        nblocks: int
            Number of blocks (default: (trajlength - corrlength) / gaplength )
        corrlength: int
            Number of frames in a block (default: trajlength /2)
        gaplength: int
            Number of frames between blocks (default: trajlength /50)
        csv_file: str
            CSV file with mean squared displacements for each blocks
        

    """

    sel = univ.select_atoms("type {}".format(atom_type))
    box = univ.dimensions

    if nblocks == None or corrlength == None or gaplength == None:
        corrlength = int(len(univ.trajectory) / 2)
        gaplength = int(len(univ.trajectory) / 50)
        if gaplength == 0:
            gaplength = 1
            print("Warning: The trajectory is too short so the gap between blocks was set to 1.")
        nblocks = int( (len(univ.trajectory) - corrlength) / gaplength )
        
    if (gaplength * nblocks + corrlength) > len(univ.trajectory):
        raise ValueError("Gap between correlation block, correlation length, or number of block too large.")

    print("Compute the profile of the mean-squared displacement on {} {} atoms...".format(len(sel), atom_type))   
    print("Total number of frames: {}".format(len(univ.trajectory)))
    print("Timestep between frames: {} fs".format(dt))
    print("Number of blocks for calculation: {}".format(nblocks))
    print("Length of blocks: {} frames / {:.2f} ps".format(
        corrlength, corrlength * dt * 1e-3))
    print("Length between origins: {} frames / {:.2f} ps\n".format(
        gaplength, gaplength * dt * 1e-3))

    if nblocks > 1:
        starts = np.arange(0, nblocks * gaplength , gaplength)
        ends = starts + corrlength
    else: 
        starts = np.array([0])
        ends = np.array([corrlength])
    
    t = np.arange(0, corrlength, 1) * dt / 1000 # time in picosecond

    msd_dic = {
    'xx': [],
    'yy': [],
    'zz': [],
    'xy': [],
    'xz': [],
    'yz': [],
    }

    for i, (start, end) in enumerate( tqdm(zip(starts, ends), total=len(starts)), 1):

        # initialize the trajectory
        univ.trajectory[start]

        # initialize positions w.r.t the center of mass to remove drift
        pos = sel.positions
        rr = pos - np.mean(pos, axis=0)

        # initialize total dr and msd
        d = np.zeros(pos.shape)
        msd_values = {key: [] for key in msd_dic.keys()}

        for ts in univ.trajectory[start:end]:

            # collect positions w.r.t the center of mass to remove the drift
            pos = sel.positions
            r = pos - np.mean(pos, axis=0)

            # calculate the differential displacement w.r.t last frame positions
            dd = r - rr

            # Taking PBC into account
            dd -= box[:3].T * (dd/box[:3].T).round()

            # Add the the differential displacement to the total displacement
            d += dd

            msd_values['xx'].append(np.mean(d[:, 0] * d[:, 0]))
            msd_values['yy'].append(np.mean(d[:, 1] * d[:, 1]))
            msd_values['zz'].append(np.mean(d[:, 2] * d[:, 2]))
            msd_values['xy'].append(np.mean(d[:, 0] * d[:, 1]))
            msd_values['xz'].append(np.mean(d[:, 0] * d[:, 2]))
            msd_values['yz'].append(np.mean(d[:, 1] * d[:, 2]))

            # save last frame position w.r.t the center of mass to remove the drift
            rr = r

        for key, values in msd_values.items():
            msd_dic[key].append(np.array(values))

    data_dic = {
        key: np.mean(values, axis=0) for key, values in msd_dic.items()
    }

    dfo = pd.DataFrame(data_dic, index = t)
    dfo.index.name = 'time'

    if csv_file is not None:
        dfo.to_csv(csv_file)

    return dfo


def msd_profile(univ: mda.Universe, 
                atom_type: str, 
                dt: int=100, 
                axis: str='z', 
                nblocks: int=None, 
                corrlength: int=100, 
                gaplength: int=100, 
                delta: float=1.0) -> tuple[pd.MultiIndex, pd.MultiIndex]:
    """Calculate a profile of diffusion coefficient in the case of liquid/solid interface from the mean squared displacement (MSD) according to Einstein's equation:

    .. math:: D = \\frac{1}{6} \\frac{\\langle [\\mathbf{r}(t_0 + \\tau) - \\mathbf{r}(t_0)]^2 \\rangle}{\\tau} = \\frac{1}{6} \\frac{\\text{MSD}}{\\tau}

    The positions of the atom selection are stored at each time origin.

    Parameters
    ----------
        univ: MDAnalysis.Universe
            The input MDAnalysis Universe to analyse
        atype: str
            The atom type to compute the diffusion coefficient
        dt: float
            Timestep between each frame of the trajectory (fs)
        axis: str
            Axis along which the profile will be calculated
        nblocks: int
            Number of blocks (default: (trajlength - corrlength) / gaplength )
        corrlength: int
            Number of frames in a block (default: trajlength /2)
        gaplength: int
            Number of frames between blocks (default: trajlength /50)
        delta: float
            Size of the spacial binning (ängstrom)

    """
    
    sel = univ.select_atoms("type {}".format(atom_type))
    box = univ.dimensions
    boxv = lattice2vectors(box)

    if nblocks == None:
        nblocks = int( (len(univ.trajectory) - corrlength) / gaplength)
   
    if (gaplength * nblocks + corrlength) > len(univ.trajectory):
        raise Exception("Gap between correlation block, correlation length, or number of block too large.")

    if nblocks > 1:
        starts = np.arange(0, nblocks * gaplength , gaplength)
        ends = starts + corrlength
    else: 
        starts = np.array([0])
        ends = np.array([corrlength])

    print("Compute the profile of the mean-squared displacement on {} {} atoms along the {} axis...".format(len(sel), atom_type, axis))   
    print("Total number of frames: {}".format(len(univ.trajectory)))
    print("Timestep between frames: {} fs".format(dt))
    print("Number of blocks: {}".format(nblocks))
    print("Length of blocks: {} frames / {} ps".format(
        corrlength, corrlength*dt*1e-3))
    print("Gap between origins: {} frames / {} ps\n".format(
        gaplength, gaplength*dt*1e-3)) 

    if axis == "x":
        axid = 0
    if axis == "y":
        axid = 1
    if axis == "z":
        axid = 2

    # if dmin == None:
    #     dmin = 0
    # if dmax == None:
    #     dmax = box[axid]

    t = np.arange(0, corrlength, 1) * dt / 1000

    # positions_array = np.array([], dtype=np.float64).reshape(0, corrlength)
    # avg_pos = np.array([], dtype=np.float64)
    # std_pos = np.array([], dtype=np.float64)

    msd_per_block = {
    'xx': [],
    'yy': [],
    'zz': [],
    'xy': [],
    'xz': [],
    'yz': [],
    }

    msd_df_dic = {
    'xx': [],
    'yy': [],
    'zz': [],
    'xy': [],
    'xz': [],
    'yz': [],
    }

    std_df_dic = {
    'xx': [],
    'yy': [],
    'zz': [],
    'xy': [],
    'xz': [],
    'yz': [],
    }

    error_df_dic = {
    'xx': [],
    'yy': [],
    'zz': [],
    'xy': [],
    'xz': [],
    'yz': [],
    }

    for i, (start, end) in enumerate( tqdm(zip(starts, ends), total=len(starts)), 1):
        
        # initialize the trajectory
        univ.trajectory[start]

        # create the selection of atoms
        sel = univ.select_atoms("type {}".format(atom_type))

        # get the reference positions at the start of the block
        rpos = sel.positions
        rposbyframe = sel.positions # to avoid problem due to PBC

        # initialize total dr and msd
        d = np.zeros(rpos.shape)
        msd_values = {key: [] for key in msd_per_block.keys()}
        positions_intra_block = []
        positions_per_block = []

        for ts in univ.trajectory[start:end]:

            # collect positions
            pos = sel.positions

            # calculate the differential displacement w.r.t last frame positions
            dd = pos - rpos

            # for i in range(3):
            #     idx = dd[:,i] > 0.5 * box[i]
            #     dd[idx] -= boxv[i]
            #     idx = -dd[:,i] > 0.5 * box[i]
            #     dd[idx] += boxv[i]

            # take PBC into account
            dd -= box[:3].T * (dd/box[:3].T).round()

            # Add the the differential displacement to the total displacement
            d += dd

            msd_values['xx'].append(d[:, 0] * d[:, 0])
            msd_values['yy'].append(d[:, 1] * d[:, 1])
            msd_values['zz'].append(d[:, 2] * d[:, 2])
            msd_values['xy'].append(d[:, 0] * d[:, 1])
            msd_values['xz'].append(d[:, 0] * d[:, 2])
            msd_values['yz'].append(d[:, 1] * d[:, 2])

            positions_intra_block.append((rposbyframe+dd)[:,axid])

            # save last frame position
            rpos = pos
            rposbyframe += dd
            
        for key, values in msd_values.items():
            msd_per_block[key].append(np.array(values).T)
            # msd_dic[key].extend(np.array(values).T)
        # positions_array = np.append(positions_array, np.array(positions_intra_block).T, axis=0)
        positions_per_block.append(np.array(positions_intra_block).T)

    msd_dic = {
        key: np.concatenate(blocks, axis=0)  # shape (n_atoms_total, corrlength)
        for key, blocks in msd_per_block.items()
    }

    positions_array = np.concatenate(positions_per_block, axis=0)
    avg_pos = np.mean(positions_array, axis=1)
    drange = np.arange(np.min(positions_array), np.max(positions_array), delta)
    binned_pos = (drange[:-1] + drange[1:]) / 2

    # Dictionnary with variation in position during the measurement of MSD
    column_groups = {i: [] for i in range(len(drange))}
    for col_index, avg in enumerate(avg_pos):
        bin_index = np.digitize(avg, drange) - 1
        if bin_index == len(drange):
            bin_index -= 1
        column_groups[bin_index].append(col_index)

    coords_var_dict = {j: positions_array[column_groups[i],:].flatten() - np.median(positions_array[column_groups[i],:].flatten()) for i, j in enumerate(np.round(binned_pos,2))}

    def remove_empty_arrays(input_dict):
        keys_to_remove = [key for key, value in input_dict.items() if value.size == 0]
        for key in keys_to_remove:
            del input_dict[key]
        return input_dict
    
    coords_var_dict = remove_empty_arrays(coords_var_dict)
        
    # create the histogram for MSD
    for key, values in msd_dic.items():
    
        binned_msd = np.array([np.mean(values[np.where((avg_pos > low) & (avg_pos <= high))], axis=0) for low, high in zip(drange[:-1], drange[1:])])
        binned_std_msd = np.array([np.std(values[np.where((avg_pos > low) & (avg_pos <= high))], axis=0) for low, high in zip(drange[:-1], drange[1:])])

        binned_pos_final = binned_pos[~np.isnan(binned_msd).any(axis=1)]
        binned_msd = binned_msd[~np.isnan(binned_msd).any(axis=1)]
        binned_std_msd = binned_std_msd[~np.isnan(binned_std_msd).any(axis=1)]

        msd_df_dic[key] = pd.DataFrame(binned_msd.T, index=t, columns=np.round(binned_pos_final,2))
        std_df_dic[key] = pd.DataFrame(binned_std_msd.T, index=t, columns=np.round(binned_pos_final,2))

        msd_df_dic[key].index.name = 'time'
        std_df_dic[key].index.name = 'time'
        
    return pd.concat(msd_df_dic, axis=1), pd.concat(std_df_dic, axis=1), coords_var_dict

def diffusion_coefficient(df_msd: pd.DataFrame, start: float=0.0) -> pd.DataFrame:
    """Calculate the diffusion coefficients for a bulk solution."""
    df_msd = df_msd.loc[start:]
    t = df_msd.index

    dc_array = []
    error_array = []

    for c in df_msd.columns:
        msd = df_msd[c]
        
        m, ci_m = _linear_fit(t, msd)
        dc_array.append(m)
        error_array.append(ci_m)

    res = pd.DataFrame([dc_array, error_array], index=['DC (m2/s)', 'Error (m2/s)'], columns=df_msd.columns)
    res *= 1e-8 / 2  # Conversion from A2/ps to m2/s
    
    dc_iso = res[['xx', 'yy', 'zz']].iloc[0].mean()
    sd_iso = np.sqrt((res[['xx', 'yy', 'zz']].iloc[1]**2).sum()) / 3

    res['3d'] = [dc_iso, sd_iso]

    return res

def diffusion_coefficient_profile(df_msd: pd.MultiIndex, 
                      df_std: pd.MultiIndex=None, 
                      start:float=1.0) -> pd.DataFrame:
    """Calculates the diffusion coefficient profile (Spatial MultiIndex)."""
    df_msd = df_msd.loc[start:]
    t = df_msd.index
    
    if df_std is not None:
        df_std = df_std.loc[start:]

    dc_array = []
    error_array = []

    components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
    for comp in components:
        msd_comp = df_msd[comp]
        std_comp = df_std[comp] if df_std is not None else None
        
        for c in msd_comp.columns:
            msd = msd_comp[c]
            std = std_comp[c] if std_comp is not None else None
            
            m, ci_m = _linear_fit(t, msd, std)
            dc_array.append(m)
            error_array.append(ci_m)

    res = pd.DataFrame([dc_array, error_array], index=['DC (m2/s)', 'Error (m2/s)'], columns=df_msd.columns)
    res *= 1e-8 / 2

    # Isotropic calculation on the MultiIndex
    dc_iso = res['xx'].iloc[0] + res['yy'].iloc[0] + res['zz'].iloc[0]
    dc_iso /= 3
    sd_iso = np.sqrt(res['xx'].iloc[1]**2 + res['yy'].iloc[1]**2 + res['zz'].iloc[1]**2) / 3
    
    iso_values = pd.DataFrame([dc_iso, sd_iso], index=res.index, columns=res['xx'].columns)
    multi_index = pd.MultiIndex.from_product([['iso'], iso_values.columns], names=df_msd.columns.names)
    iso_values.columns = multi_index
    
    res = pd.concat([res, iso_values], axis=1)
    return res

# def diffusion(df_msd, df_std=None, start=None) -> pd.DataFrame:
#     """Calculate the diffusion from a DataFrame of MSD and a DataFrame of standard deviation on the MSD using a weighted linear regression. The DataFrame can be with a single index like obtained from msd_bulk or with multiple index like obtained from msd_profile.

#      Parameters
#     ----------
#         df_msd: DataFrame
#             DataFrame containing the MSD.
#         df_std: DataFrame
#             DataFrame containing the standard deviation on the MSD
#         start: float
#             Time at which the linear regression starts (ps)
#     """

#     if start is None:
#         start = 0

#     df_msd = df_msd.loc[start:]
#     t = df_msd.index
    
#     dc_array = []
#     error_array = []

#     # MSD profile (for interfaces)
#     if isinstance(df_msd.columns, pd.MultiIndex):

#         df_std = df_std.loc[start:]

#         components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
#         for comp in components:
#             msd_comp = df_msd[comp]
#             std_comp = df_std[comp]
#             for c in msd_comp.columns:
#                 msd = msd_comp[c]
#                 std = std_comp[c]
#                 m, ci_m = _linear_fit(msd, std)
#                 dc_array.append(m)
#                 error_array.append(ci_m)

#         # Calculate diffusion parallel to the surface
#         dc_para = (res.xx.iloc[0] + res.yy.iloc[0]) / 2
#         sd_para = np.sqrt(res.xx.iloc[1]**2 + res.yy.iloc[1]**2) / 2
#         para_values = pd.DataFrame([dc_para, sd_para], index=res.index, columns=res.xx.columns)
#         multi_index = pd.MultiIndex.from_product([['para'], para_values.columns], names=para_values.index)
#         para_values.columns = multi_index
#         res = pd.concat([res, para_values], axis=1)

#         res = pd.DataFrame([dc_array, error_array], index=['DC (m2/s)', 'Error (m2/s)'], columns=df_msd.columns)
#         res *= 1e-8 / 2 # from A2/ps to m2/s
#         dc_iso = (res.xx.iloc[0] + res.yy.iloc[0] + res.zz.iloc[0]) / 3 # we divide per 3 for the dimensionality of the isotropic diffusion coefficient (3D)
#         sd_iso = np.sqrt(res.xx.iloc[1]**2 + res.yy.iloc[1]**2 + res.zz.iloc[1]**2) / 3
#         iso_values = pd.DataFrame([dc_iso, sd_iso], index=res.index, columns=res.xx.columns)
#         multi_index = pd.MultiIndex.from_product([['iso'], iso_values.columns], names=iso_values.index)
#         iso_values.columns = multi_index
#         res = pd.concat([res, iso_values], axis=1)
    
#     else:
#         for c in df_msd.columns:
#             msd = df_msd[c]
#             m, ci_m = _linear_fit(msd)
#             dc_array.append(m)
#             error_array.append(ci_m)

#         res = pd.DataFrame([dc_array, error_array], index=['DC (m2/s)', 'Error (m2/s)'], columns=df_msd.columns)
#         res *= 1e-8 / 2 # from A2/ps to m2/s

#         # Calculate isotropic diffusion
#         dc_iso = (res.xx.iloc[0] + res.yy.iloc[0] + res.zz.iloc[0]) / 3 # we divide per 3 for the dimensionality of the isotropic diffusion coefficient (3D)
#         sd_iso = np.sqrt(res.xx.iloc[1]**2 + res.yy.iloc[1]**2 + res.zz.iloc[1]**2) / 3
#         iso_values = pd.DataFrame([dc_iso, sd_iso], index=res.index, columns=res.xx.columns)
#         multi_index = pd.MultiIndex.from_product([['iso'], iso_values.columns], names=iso_values.index)
#         iso_values.columns = multi_index
#         res = pd.concat([res, iso_values], axis=1)

#         # res = pd.concat([res.T, new_row], axis=1)

#     return res

def plot_msd(df_msd: pd.DataFrame) -> None:
    """Plot the MSD from a pandas DataFrame."""

    # Plot using Seaborn
    half_size = int(len(df_msd.columns)/2)

    # Get the colormaps
    cmap = plt.get_cmap("coolwarm")
    cmap_r = plt.get_cmap("coolwarm_r")
    
    # Generate arrays of values from 0.0 to 1.0 to sample colors
    colors_forward = cmap(np.linspace(0, 1, half_size))
    
    if len(df_msd.columns) % 2 == 0:
        colors_reverse = cmap_r(np.linspace(0, 1, half_size))
    else:
        colors_reverse = cmap_r(np.linspace(0, 1, int(half_size + 1)))
        
    # Combine the lists (Matplotlib handles RGBA arrays natively, no need for hex string conversion)
    palettef = np.vstack([colors_forward, colors_reverse])

    # Dashes
    dashes = []
    for i in range(len(df_msd.columns)):
        if i == 0 or i == len(df_msd.columns) - 1:
            dashes.append('--')
        else:
            dashes.append('-')   

    fig, ax = plt.subplots()
    for i, (msd, col) in enumerate(zip(df_msd.values.T, df_msd.columns)):
        ax.plot(df_msd.index, msd, color=palettef[i], linestyle=dashes[i], label=np.round(col,2))
        
    # Set labels and title
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('MSD ($\AA^2$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()

    # Legend labels
    num_cols = int(np.ceil(len(df_msd.columns)/20))
    ax.legend(title='z ($\AA$)', bbox_to_anchor=(1.04, 0.5), loc='center left', ncols=num_cols)

    # Adjust the plot layout to accommodate the legend
    fig.subplots_adjust(right=1-num_cols*0.20)

    # Show the plot
    fig.tight_layout()
    fig.show()

def plot_diffusion_profile(df_dc: pd.DataFrame) -> None:
    """Plot the diffusion profile from a pandas DataFrame."""

    if df_dc.shape[1] > df_dc.shape[0]:
        df_dc = df_dc.T

    t = df_dc.index
    dc = df_dc['DC (m2/s)']
    error = df_dc['Error (m2/s)']

    plt.scatter(t, dc, color='royalblue')
    plt.fill_between(t, dc-error, dc+error, alpha=0.2, color='royalblue')

    imax = dc.argmax()

    # Set labels and title
    plt.xlabel('Distance ($\AA$)')
    plt.ylabel('Diffusion coefficient ($m^2/s$)')

    plt.ylim(dc.min(), dc.max()+dc.iloc[imax])
    plt.grid()
    plt.show()
    
