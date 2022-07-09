from dask.distributed import Client, SSHCluster
import sys
import dask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
import matplotlib.pyplot as plt

from dask import delayed
from time import time

# ------ ALGORITHM FUNCTIONS
def load(path):
    return np.load(path, allow_pickle=True)["a"]

def R_yz(theta_rot, phi_rot):
    return np.array([[np.cos(phi_rot) * np.cos(theta_rot), -np.sin(phi_rot) * np.cos(theta_rot), np.sin(theta_rot)], 
                     [np.sin(phi_rot), np.cos(phi_rot), 0], 
                     [-np.sin(theta_rot) * np.cos(phi_rot), np.sin(theta_rot) * np.sin(phi_rot), np.cos(theta_rot)]])

def convert_pmt_ids(input_ids, conversion_ids):
    cd_ids  = np.array(conversion_ids["CdID"])
    pmt_ids = np.array(conversion_ids["PMTID"])
    mask    = np.isin(cd_ids, input_ids)
    return pmt_ids[mask]

def find_pmt_coord(pmt_positions, data_pmt_id):
    return pmt_positions[
        np.isin(pmt_positions.PMTID, data_pmt_id)
        ].loc[:, ["x", "y", "z"]].reset_index(drop=True).to_numpy()

# ------ DASK FUNCTIONS
def load_bag(path, Nevents):
    data_np = load(path)
    data_np = data_np[:, :Nevents]
    return [np.vstack([ data_np[j, i] for j in range(3)]) for i in range(data_np.shape[1])]


# ------ DISTRIBUTED ALGORITHM FUNCTIONS
def rotate_ev(data):

    nonzeros_inds = data[2] != 0.0
    data_pmt_id   = convert_pmt_ids(data[0][nonzeros_inds], conversion_ids)
    pmt_coord     = find_pmt_coord(pmt_positions, data_pmt_id)

    tot_charge = sum(data[1][nonzeros_inds])
    x_cc       = sum(pmt_coord[:,0] * data[1][nonzeros_inds]) / tot_charge
    y_cc       = sum(pmt_coord[:,1] * data[1][nonzeros_inds]) / tot_charge
    z_cc       = sum(pmt_coord[:,2] * data[1][nonzeros_inds]) / tot_charge

    theta_cc   = np.arctan2(
        np.sqrt((x_cc)**2+(y_cc)**2), z_cc
    )
    phi_cc     = np.arctan2(y_cc, x_cc) 

    theta_rot = -theta_cc + np.pi/2
    phi_rot   = -phi_cc
    
    # coord_new = np.matmul(R_yz(theta_rot, phi_rot), pmt_coord.T)
    coord_new = np.matmul(
        R_yz(theta_rot, phi_rot), pmt_coord.T
    )

    R = np.sqrt(np.sum(np.power(coord_new, 2), axis=0))

    charge_hitt = np.vstack([data[1], data[2]])
    charge_hitt = charge_hitt[:,nonzeros_inds]

    rotated = np.vstack([coord_new, R, charge_hitt])
    
    del coord_new
    del charge_hitt
    del pmt_coord
    
    return rotated

def mapping_single_event(rotated_ev):
    ####################
    # rotated_ev must be computed
    ####################
    
    N_max = 115
    
    coord_new   = rotated_ev[:3]
    charge_hitt = rotated_ev[4:, ].T
    R           = rotated_ev[3, ].mean()

    z_levels, step = np.linspace(coord_new[2,].min(), coord_new[2,].max(), 124, retstep=True)
    #z_levels       = z_levels.persist()
    image_mat      = np.zeros((230,124,2))

    #masks = 

    for j, z in enumerate(z_levels):
        mask = (np.abs(coord_new[2,] - z) < step)         #(np.abs(pmt_pos.z - z) < delta)
        if(not np.any(mask)): continue
        masked = coord_new[:,mask]


        Rz = (R**2 - z**2)
        Neff = 0 if Rz < 0 else N_max * np.sqrt(Rz) / R
        #ix = np.zeros(np.sum(mask), dtype=np.int32)
        ix = np.around( Neff * (np.arctan2(masked[1,], masked[0,]) / np.pi) + (N_max / 2) ) + 57
        ix = ix.astype(np.int32)
        #ix = ix.compute()
        if(np.any(ix >= 230)):
            ix[ix >= 230] = ix[ix >= 230] - 230

        image_mat[ix, j,] = charge_hitt[mask, ]

                # if np.isnan(mat[ix, i+1]):
                #     mat[ix, i+1] = row.id
                # else:
                #     mat[ix, 123 if i else i] = row.id

    del rotated_ev
    return image_mat


###############################################################################################################
data_folder       = "/root/data/data/real/train/data/"
pmt_pos_fname     = "/root/distributed-juno/cluster/PMTPos_CD_LPMT.csv"
pmt_id_conv_fname = "/root/distributed-juno/cluster/PMT_ID_conversion.csv"
train_data_fname  = "raw_data_train_4.npz"


pmt_positions     = pd.read_csv(pmt_pos_fname)
pmt_id_conversion = pd.read_csv(pmt_id_conv_fname)
conversion_ids    = pd.read_csv(pmt_id_conv_fname)


w_max      = 8
workers    = np.arange(1, w_max+1)
threads    = np.arange(1, w_max+1)
partitions = [1, 2, 4, 8, 16, 32, 64]

Nevents    = 1024

ips        = ["10.67.22.39", "10.67.22.74", "10.67.22.27", "10.67.22.91", "10.67.22.60"]
ssh_opt    = {"known_hosts": "/root/.ssh/known_hosts"}
sched_port = {"dashboard_address": ":8787"}
# worker_par = {"nthreads": 1, "n_workers": 4}
###############################################################################################################



###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
worker_list        = []
thread_list        = []
partition_list     = []
load_time_list     = []
rotation_time_list = []
mapping_time_list  = []
compute_time_list  = []
total_time_list    = []

# for n_p in partitions:
for n_w in workers:
    for n_t in threads:
            
            # n_t = 1 # hardcoding threads
            n_p = 4*n_w # hardcode partitions
            
            worker_par = {"nthreads": int(n_t), "n_workers": int(n_w)}

            cluster = SSHCluster(
                ips,
                connect_options   = ssh_opt,
                worker_options    = worker_par,
                scheduler_options = sched_port
            )
            client = Client(cluster)
#           print(client)

            print(f"\n\nPROCESSING: {n_w} workers {n_t} threads and {n_p} partitions\n")
            #################################
            time_0  = time()

            start   = time()
            data_db = db.from_sequence(load_bag(data_folder + train_data_fname, Nevents=Nevents), npartitions=n_p)
            stop    = time()

            load_time = stop-start
            print("Load time:\t", load_time)

            start   = time()
            rotated = db.map(rotate_ev, data_db)
            stop    = time()

            rotation_time = stop-start
#            print("Rotation time:\t", rotation_time)

            start   = time()
            mapped  = db.map(mapping_single_event, rotated)
            stop    = time()

            projection_time = stop-start
#            print("Mapping time:\t", projection_time)

            start   = time()
            images  = mapped.compute()
            stop    = time()

            compute_time = stop-start
            print("Compute time:\t", compute_time)



            time_1 = time()
            total_time = time_1 - time_0
            print("\n\nTotal time:\t", total_time)
            #################################

            client.shutdown()
            client.close()

            worker_list.append(n_w)        
            thread_list.append(n_t)   
            partition_list.append(n_p)
            load_time_list.append(load_time)     
            rotation_time_list.append(rotation_time) 
            mapping_time_list.append(projection_time)  
            compute_time_list.append(compute_time)  
            total_time_list.append(total_time)   
        
        
grid_results = pd.DataFrame(
    {
        "n_workers"     : worker_list,
        "n_threads"     : thread_list,
        "n_partitions"  : partition_list,
        "load_time"     : load_time_list,
        "rotation_time" : rotation_time_list,
        "mapping_time"  : mapping_time_list,
        "compute_time"  : compute_time_list,
        "total_time"    : total_time_list,
    }
)
grid_results.to_csv("grid_results_wtp.csv", index=False)
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################