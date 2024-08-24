import os
import glob

import pandas as pd
import pcdl


FEATURES_COLS = [
    'time',
    'cell_type',
    'position_x',
    'position_y',
    # 'position_z',
    'total_volume',
    # 'nuclear_volume',
    # 'fluid_fraction',
    # 'calcified_fraction',
    # 'cell_density_micron3',
]


def read_physicell_output(physicell_out_folder):
    physicell_sim = pcdl.TimeSeries(physicell_out_folder)
    return physicell_sim.get_mcds_list()


def merge_physicell_output(mcds_list):
    mcds_df = pd.DataFrame([])
    for timestep in mcds_list:
        df_cell = timestep.get_cell_df(states=1)
        df_cell = df_cell[FEATURES_COLS].reset_index()
        mcds_df = pd.concat([mcds_df, df_cell], axis=0)
    mcds_df = mcds_df.sort_values(by=["ID", "time"])
    mcds_df = mcds_df.rename(index=str, columns={"ID": "cell_id"})
    mcds_df = mcds_df.reset_index(drop=True)
    return mcds_df


def get_physicell_df(physicell_out_folder):
    mcds_list = read_physicell_output(physicell_out_folder)
    mcds_df = merge_physicell_output(mcds_list)
    return mcds_df


def merge_physicell_trajectories(data_dir, project):
    full_df = pd.DataFrame()
    for filename in glob.glob(os.path.join(data_dir, project, "raw", f"{project}*.csv")):
        mcds_df = pd.read_csv(filename)
        mcds_df = mcds_df[["cell_id"] + FEATURES_COLS]
        mcds_df["cell_type"] = mcds_df["cell_type"].astype("category").cat.codes
        mcds_df["traj"] = filename.split("/")[-1].replace(".csv", "").split("_")[-1]
        mcds_df = mcds_df.astype("int32")
        full_df = pd.concat([full_df, mcds_df], axis=0)
    full_df = full_df.reset_index(drop=True)
    return full_df
