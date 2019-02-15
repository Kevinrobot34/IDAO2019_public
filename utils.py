# Some updates are made by Team BarelyBears

import os
from itertools import repeat
import numpy as np
import pandas as pd
import tables

HDF_PATHS = {}
HDF_PATHS['trn_simple'] = '01_rawdata/trn_data_simple.hdf'
HDF_PATHS['trn_foi'] = '01_rawdata/trn_data_foi.hdf'
HDF_PATHS['trn_closesthit'] = '01_rawdata/trn_data_closest_hit_features.hdf'
HDF_PATHS['trn_averagehit'] = '01_rawdata/trn_data_average_hit_features.hdf'
HDF_PATHS['trn_hitcount'] = '01_rawdata/trn_data_hit_count_features.hdf'
HDF_PATHS['trn_angle'] = '01_rawdata/trn_data_angle_features.hdf'
HDF_PATHS['trn_misc'] = '01_rawdata/trn_data_misc.hdf'
HDF_PATHS['test_simple'] = '01_rawdata/test_data_simple.hdf'
HDF_PATHS['test_foi'] = '01_rawdata/test_data_foi.hdf'
HDF_PATHS['test_closesthit'] = '01_rawdata/test_data_closest_hit_features.hdf'
HDF_PATHS['test_averagehit'] = '01_rawdata/test_data_average_hit_features.hdf'
HDF_PATHS['test_hitcount'] = '01_rawdata/test_data_hit_count_features.hdf'
HDF_PATHS['test_angle'] = '01_rawdata/test_data_angle_features.hdf'

SIMPLE_FEATURE_COLUMNS = ['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]',
       'avg_cs[1]', 'avg_cs[2]', 'avg_cs[3]', 'ndof', 'MatchedHit_TYPE[0]',
       'MatchedHit_TYPE[1]', 'MatchedHit_TYPE[2]', 'MatchedHit_TYPE[3]',
       'MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]', #13
       'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
       'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
       'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]',
       'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]',
       'MatchedHit_DX[3]', 'MatchedHit_DY[0]', 'MatchedHit_DY[1]',
       'MatchedHit_DY[2]', 'MatchedHit_DY[3]', 'MatchedHit_DZ[0]',
       'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]',
       'MatchedHit_T[0]', 'MatchedHit_T[1]', 'MatchedHit_T[2]',
       'MatchedHit_T[3]', 'MatchedHit_DT[0]', 'MatchedHit_DT[1]',
       'MatchedHit_DT[2]', 'MatchedHit_DT[3]', 'Lextra_X[0]', 'Lextra_X[1]',
       'Lextra_X[2]', 'Lextra_X[3]', 'Lextra_Y[0]', 'Lextra_Y[1]',
       'Lextra_Y[2]', 'Lextra_Y[3]', 'NShared', 'Mextra_DX2[0]',
       'Mextra_DX2[1]', 'Mextra_DX2[2]', 'Mextra_DX2[3]', 'Mextra_DY2[0]',
       'Mextra_DY2[1]', 'Mextra_DY2[2]', 'Mextra_DY2[3]', 'FOI_hits_N', 'PT', 'P']
FOI_COLUMNS = ["FOI_hits_X", "FOI_hits_Y", "FOI_hits_T",
               "FOI_hits_Z", "FOI_hits_DX", "FOI_hits_DY", "FOI_hits_S"]
CLOSEST_HIT_FEATURE_COLUMNS = ['closest_x_per_station[0]', 'closest_x_per_station[1]', #65-
       'closest_x_per_station[2]', 'closest_x_per_station[3]',
       'closest_y_per_station[0]', 'closest_y_per_station[1]',
       'closest_y_per_station[2]', 'closest_y_per_station[3]',
       'closest_T_per_station[0]', 'closest_T_per_station[1]',
       'closest_T_per_station[2]', 'closest_T_per_station[3]',
       'closest_z_per_station[0]', 'closest_z_per_station[1]',
       'closest_z_per_station[2]', 'closest_z_per_station[3]',
       'closest_dx_per_station[0]', 'closest_dx_per_station[1]',
       'closest_dx_per_station[2]', 'closest_dx_per_station[3]',
       'closest_dy_per_station[0]', 'closest_dy_per_station[1]',
       'closest_dy_per_station[2]', 'closest_dy_per_station[3]']
AVERAGE_HIT_COLUMNS = ["average_x_per_station[0]", "average_x_per_station[1]", "average_x_per_station[2]", "average_x_per_station[3]", #89-
"average_y_per_station[0]", "average_y_per_station[1]", "average_y_per_station[2]", "average_y_per_station[3]"]
HIT_COUNT_COLUMNS = ["FOI_hits_S_0", "FOI_hits_S_1", "FOI_hits_S_2", "FOI_hits_S_3"] #97-
ANGLE_COLUMNS = ['Pangle', 'MAngle[0]', 'MAngle[1]'] #101-
TRAIN_COLUMNS = ["label", "weight"]
ID_COLUMN = "id"
FULL_COLUMNS = SIMPLE_FEATURE_COLUMNS + CLOSEST_HIT_FEATURE_COLUMNS + AVERAGE_HIT_COLUMNS + HIT_COUNT_COLUMNS + ANGLE_COLUMNS

COL_ESSENTIAL = SIMPLE_FEATURE_COLUMNS + ['closest_x_per_station[0]',
       'closest_x_per_station[1]', 'closest_x_per_station[2]',
       'closest_x_per_station[3]', 'closest_y_per_station[0]',
       'closest_y_per_station[1]', 'closest_y_per_station[2]',
       'closest_y_per_station[3]', 'Pangle', 'average_x_per_station[0]',
       'average_x_per_station[1]', 'average_x_per_station[2]',
       'average_x_per_station[3]', 'average_y_per_station[0]',
       'average_y_per_station[1]', 'average_y_per_station[2]',
       'average_y_per_station[3]', 'MAngle[0]', 'MAngle[1]', 'MAngle',
       'MAngle_v2[0]', 'MAngle_v2[1]', 'MAngle_v2[2]',
       'closest_xy_per_station[0]', 'closest_xy_per_station[1]',
       'closest_xy_per_station[2]', 'closest_xy_per_station[3]']

N_STATIONS = 4
FEATURES_PER_STATION = 6
N_FOI_FEATURES = N_STATIONS*FEATURES_PER_STATION
# The value to use for stations with missing hits
# when computing FOI features
EMPTY_FILLER = np.nan

# Examples on working with the provided files in different ways

# hdf is all fine - but it requires unpickling the numpy arrays
# which is not guranteed
def load_hdf(paths):
    return pd.concat([pd.read_hdf(HDF_PATHS[path]) for path in paths], axis = 1)

def load_train_hdf(path):
    return pd.concat([
        pd.read_hdf(os.path.join(path, "train_part_%i_v2.hdf" % i))
        for i in (1, 2)], axis=0, ignore_index=True)


def load_data_csv(path, feature_columns, dtype = None):
    train = pd.concat([
        pd.read_csv(os.path.join(path, "train_part_%i_v2.csv.gz" % i),
                    usecols= [ID_COLUMN] + feature_columns + TRAIN_COLUMNS,
                    index_col=ID_COLUMN, dtype = dtype)
        for i in (1, 2)], axis=0, ignore_index=True)
    test = pd.read_csv(os.path.join(path, "test_public_v2.csv.gz"),
                       usecols=[ID_COLUMN] + feature_columns, index_col=ID_COLUMN, dtype = dtype)
    return train, test


def parse_array(line, dtype=np.float32):
    return np.fromstring(line[1:-1], sep=" ", dtype=dtype)

def parse_row(row):
    return row.apply(parse_array)

def load_full_test_csv(path):
    converters = dict(zip(FOI_COLUMNS, repeat(parse_array)))
    types = dict(zip(SIMPLE_FEATURE_COLUMNS, repeat(np.float32)))
    test = pd.read_csv(os.path.join(path, "test_public_v2.csv.gz"),
                       index_col="id", converters=converters,
                       dtype=types,
                       usecols=[ID_COLUMN]+SIMPLE_FEATURE_COLUMNS+FOI_COLUMNS)
    return test

def find_closest_hit_per_station(row, parse=True):
    result = np.empty(N_FOI_FEATURES, dtype=np.float32)
    closest_x_per_station = result[0:4]
    closest_y_per_station = result[4:8]
    closest_T_per_station = result[8:12]
    closest_z_per_station = result[12:16]
    closest_dx_per_station = result[16:20]
    closest_dy_per_station = result[20:24]

    var = {}
    for col in FOI_COLUMNS:
        if parse:
            var[col] = parse_array(row[col])
        else:
            var[col] = row[col]

    for station in range(4):
        hits = (var["FOI_hits_S"] == station)
        if not hits.any():
            closest_x_per_station[station] = EMPTY_FILLER
            closest_y_per_station[station] = EMPTY_FILLER
            closest_T_per_station[station] = EMPTY_FILLER
            closest_z_per_station[station] = EMPTY_FILLER
            closest_dx_per_station[station] = EMPTY_FILLER
            closest_dy_per_station[station] = EMPTY_FILLER
        else:
            x_distances_2 = (row["Lextra_X[%i]" % station] - var["FOI_hits_X"][hits])**2
            y_distances_2 = (row["Lextra_Y[%i]" % station] - var["FOI_hits_Y"][hits])**2
            distances_2 = x_distances_2 + y_distances_2
            closest_hit = np.argmin(distances_2)
            closest_x_per_station[station] = x_distances_2[closest_hit]
            closest_y_per_station[station] = y_distances_2[closest_hit]
            closest_T_per_station[station] = var["FOI_hits_T"][hits][closest_hit]
            closest_z_per_station[station] = var["FOI_hits_Z"][hits][closest_hit]
            closest_dx_per_station[station] = var["FOI_hits_DX"][hits][closest_hit]
            closest_dy_per_station[station] = var["FOI_hits_DY"][hits][closest_hit]
    return result

def add_foi_features(row, parse=True):
    result = np.empty(len(CLOSEST_HIT_FEATURE_COLUMNS + AVERAGE_HIT_COLUMNS + HIT_COUNT_COLUMNS), dtype=np.float32)
    closest_x_per_station = result[0:4]
    closest_y_per_station = result[4:8]
    closest_T_per_station = result[8:12]
    closest_z_per_station = result[12:16]
    closest_dx_per_station = result[16:20]
    closest_dy_per_station = result[20:24]
    average_x_per_station = result[24:28]
    average_y_per_station = result[28:32]
    hit_count_per_station = result[32:36]

    var = {}
    for col in FOI_COLUMNS:
        if parse:
            var[col] = parse_array(row[col])
        else:
            var[col] = row[col]

    for station in range(4):
        hits = (var["FOI_hits_S"] == station)
        hit_count_per_station[station] = hits.sum()
        if not hits.any():
            closest_x_per_station[station] = EMPTY_FILLER
            closest_y_per_station[station] = EMPTY_FILLER
            closest_T_per_station[station] = EMPTY_FILLER
            closest_z_per_station[station] = EMPTY_FILLER
            closest_dx_per_station[station] = EMPTY_FILLER
            closest_dy_per_station[station] = EMPTY_FILLER
            average_x_per_station[station] = EMPTY_FILLER
            average_y_per_station[station] = EMPTY_FILLER
        else:
            x_distances_2 = (row["Lextra_X[%i]" % station] - var["FOI_hits_X"][hits])**2
            y_distances_2 = (row["Lextra_Y[%i]" % station] - var["FOI_hits_Y"][hits])**2
            distances_2 = x_distances_2 + y_distances_2
            closest_hit = np.argmin(distances_2)
            closest_x_per_station[station] = x_distances_2[closest_hit]
            closest_y_per_station[station] = y_distances_2[closest_hit]
            closest_T_per_station[station] = var["FOI_hits_T"][hits][closest_hit]
            closest_z_per_station[station] = var["FOI_hits_Z"][hits][closest_hit]
            closest_dx_per_station[station] = var["FOI_hits_DX"][hits][closest_hit]
            closest_dy_per_station[station] = var["FOI_hits_DY"][hits][closest_hit]
            average_x_per_station[station] = np.average(x_distances_2)
            average_y_per_station[station] = np.average(y_distances_2)
    return result
