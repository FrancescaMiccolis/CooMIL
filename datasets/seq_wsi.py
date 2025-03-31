from __future__ import print_function, division
from asyncio import base_tasks
import os
import torch
import numpy as np
import joblib
import glob 
import tqdm
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
import collections
from itertools import islice
import bisect
import torch.nn.functional as f
from torch.utils.data import Dataset
import h5py

import torch.nn.functional as F
from torch.utils.data import DataLoader
# from cl_wsi import datasets
from utils.conf import base_path
# from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from backbone.model_clam import CLAM_SB
from backbone.hit import HIT
from backbone.transmil import TransMIL
from backbone.dsmil import FCLayer, BClassifier, MILNet
import random
from sklearn.model_selection import train_test_split

def seed_worker(worker_id):
    # Generate a seed for the worker based on the initial seed
    worker_seed = torch.initial_seed() % 2**32
    # Set the seed for NumPy operations in the worker
    np.random.seed(worker_seed)
    # Set the seed for random number generation in the worker
    random.seed(worker_seed)
    
def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    img2 = torch.cat([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2] for item in batch])
    if len(batch[0]) > 3:
        logits = torch.cat([item[3] for item in batch], dim = 0)
        return [img, img2, label, logits]
    return [img, img2, label]

# from utils.utils import generate_split, nth

def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
    seed = 7, label_frac = 1.0, custom_test_ids = None):
    indices = np.arange(samples).astype(int)
    
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None: # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None: # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            
            else:
                sample_num  = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids

def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator,n, None), default)


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

    df.to_csv(filename)
    print()
class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self, csv_path='/homes/gbontempo/continual-MIL/data/mil', name="Elephant", args=None):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            name: name of the dataset
            args: arguments
        """
        self.num_classes = 2
        self.seed = args.seed
        self.name = name
        self.args = args
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        if len(csv_path) > 0:
            self.slide_data = pd.read_csv(csv_path)
 

    def __len__(self):
        return self.slide_data.shape[0]
    
    def reset_label(self,data):
        if self.args.cam=="reverse_order":
            data.labels= abs(data.labels-7)
        if self.args.cam=="brca":
            data.labels= data.labels-2
        return data

    def return_splits(self):
        data = pd.DataFrame(self.slide_data)
        data=self.reset_label(data)
        if self.args.debug_mode:
            train_ids, val_ids, test_ids = self.generate_split_debug(data)
        else:
            train_ids, val_ids, test_ids = self.generate_split(data, self.args.fold[0], self.seed)
        assert len(set(train_ids).intersection(val_ids)) == 0
        assert len(set(val_ids).intersection(test_ids)) == 0

        train_dataset = Generic_Split(data.iloc[train_ids], data_dir=self.data_dir, args=self.args)
        val_dataset = Generic_Split(data.iloc[val_ids], data_dir=self.data_dir, args=self.args)
        test_dataset = Generic_Split(data.iloc[test_ids], data_dir=self.data_dir, args=self.args)
        return train_dataset, val_dataset, test_dataset

    def __getitem__(self, idx):
        return None

    def generate_split(self, data, fold, seed):

        # test id if fold is one of the first three folds
        #test_filter = data["fold"] < 3
        #val_filter = data["fold"] == fold
        #train_filter = (data["fold"] > 3) & (data["fold"] != fold)

        ids = list(range(10))
        ids.remove(self.args.fold[0])
        val_id = np.random.choice(ids,1)[0]
        print('Val fold id:',val_id)
        print('Test fold id:',self.args.fold[0])
        test_filter = data["fold"] == self.args.fold[0]
        val_filter = data["fold"] == val_id
        train_filter = (data["fold"] != self.args.fold[0]) & (data["fold"] != val_id)

        test_ids = data[test_filter].index
        val_ids = data[val_filter].index
        train_ids = data[train_filter].index
        return train_ids, val_ids, test_ids


    def split_indices(self, df):
        train_idx, temp_idx = train_test_split(df.index, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        return train_idx, val_idx, test_idx

    def generate_split_debug(self, data):
        vals = np.unique(data['labels'].values)
        class_A = data[self.slide_data['labels'] == vals[0]][:30]
        class_B = data[self.slide_data['labels'] == vals[1]][:30]
        train_ids_A, val_ids_A, test_ids_A = self.split_indices(class_A)
        train_ids_B, val_ids_B, test_ids_B = self.split_indices(class_B)

        train_ids = list(train_ids_A) + list(train_ids_B)
        val_ids = list(val_ids_A) + list(val_ids_B)
        test_ids = list(test_ids_A) + list(test_ids_B)

        return train_ids, val_ids, test_ids
        


# class Generic_WSI_Classification_Dataset(Dataset):
#     def __init__(self,
#         csv_path = 'dataset_csv/ccrcc_clean.csv',
#         shuffle = False, 
#         seed = 7, 
#         print_info = True,
#         label_dict = {},
#         filter_dict = {},
#         ignore=[],
#         patient_strat=False,
#         label_col = None,
#         patient_voting = 'max',
#         ):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             shuffle (boolean): Whether to shuffle
#             seed (int): random seed for shuffling the data
#             print_info (boolean): Whether to print a summary of the dataset
#             label_dict (dict): Dictionary with key, value pairs for converting str labels to int
#             ignore (list): List containing class labels to ignore
#         """
#         self.label_dict = label_dict
#         self.num_classes = len(set(self.label_dict.values()))
#         self.seed = seed
#         self.print_info = print_info
#         self.patient_strat = patient_strat
#         self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
#         self.data_dir = None
#         if not label_col:
#             label_col = 'label'
#         self.label_col = label_col

#         csv_path="/work/H2020DeciderFicarra/fmiccolis/miccai_2025_workshop/continual-MIL/lung10fold_conch.csv"
#         slide_data = pd.read_csv(csv_path)
#         slide_data = self.filter_df(slide_data, filter_dict)
#         slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

#         ###shuffle data
#         if shuffle:
#             np.random.seed(seed)
#             np.random.shuffle(slide_data)

#         self.slide_data = slide_data

#         self.patient_data_prep(patient_voting)
#         self.cls_ids_prep()

#         if print_info:
#             self.summarize()

#     def cls_ids_prep(self):
#         # store ids corresponding each class at the patient or case level
#         self.patient_cls_ids = [[] for i in range(self.num_classes)]		
#         for i in range(self.num_classes):
#             self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

#         # store ids corresponding each class at the slide level
#         self.slide_cls_ids = [[] for i in range(self.num_classes)]
#         for i in range(self.num_classes):
#             self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

#     def patient_data_prep(self, patient_voting='max'):
#         patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
#         patient_labels = []
        
#         for p in patients:
#             locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
#             assert len(locations) > 0
#             label = self.slide_data['label'][locations].values
#             if patient_voting == 'max':
#                 label = label.max() # get patient label (MIL convention)
#             elif patient_voting == 'maj':
#                 label = stats.mode(label)[0]
#             else:
#                 raise NotImplementedError
#             patient_labels.append(label)
        
#         self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

#     @staticmethod
#     def df_prep(data, label_dict, ignore, label_col):
#         if label_col != 'label':
#             data['label'] = data[label_col].copy()

#         mask = data['label'].isin(ignore)
#         data = data[~mask]
#         data.reset_index(drop=True, inplace=True)
#         for i in data.index:
#             key = data.loc[i, 'label']
#             data.at[i, 'label'] = label_dict[key]

#         return data

#     def filter_df(self, df, filter_dict={}):
#         if len(filter_dict) > 0:
#             filter_mask = np.full(len(df), True, bool)
#             # assert 'label' not in filter_dict.keys()
#             for key, val in filter_dict.items():
#                 mask = df[key].isin(val)
#                 filter_mask = np.logical_and(filter_mask, mask)
#             df = df[filter_mask]
#         return df

#     def __len__(self):
#         if self.patient_strat:
#             return len(self.patient_data['case_id'])

#         else:
#             return len(self.slide_data)

#     def summarize(self):
#         print("label column: {}".format(self.label_col))
#         print("label dictionary: {}".format(self.label_dict))
#         print("number of classes: {}".format(self.num_classes))
#         print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
#         for i in range(self.num_classes):
#             print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
#             print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

#     def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
#         settings = {
#                     'n_splits' : k, 
#                     'val_num' : val_num, 
#                     'test_num': test_num,
#                     'label_frac': label_frac,
#                     'seed': self.seed,
#                     'custom_test_ids': custom_test_ids
#                     }

#         if self.patient_strat:
#             settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
#         else:
#             settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

#         self.split_gen = generate_split(**settings)

#     def set_splits(self,start_from=None):
#         if start_from:
#             ids = nth(self.split_gen, start_from)

#         else:
#             ids = next(self.split_gen)

#         if self.patient_strat:
#             slide_ids = [[] for i in range(len(ids))] 

#             for split in range(len(ids)): 
#                 for idx in ids[split]:
#                     case_id = self.patient_data['case_id'][idx]
#                     slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
#                     slide_ids[split].extend(slide_indices)

#             self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

#         else:
#             self.train_ids, self.val_ids, self.test_ids = ids

#     def get_split_from_df(self, all_splits, split_key='train'):
#         split = all_splits[split_key]
#         split = split.dropna().reset_index(drop=True)
#         # import ipdb;ipdb.set_trace()

#         if len(split) > 0:
#             mask = self.slide_data['slide_id'].isin(split.tolist())
#             df_slice = self.slide_data[mask].reset_index(drop=True)
#             split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
#         else:
#             split = None
        
#         return split

#     def get_merged_split_from_df(self, all_splits, split_keys=['train']):
#         merged_split = []
#         for split_key in split_keys:
#             split = all_splits[split_key]
#             split = split.dropna().reset_index(drop=True).tolist()
#             merged_split.extend(split)

#         if len(split) > 0:
#             mask = self.slide_data['slide_id'].isin(merged_split)
#             df_slice = self.slide_data[mask].reset_index(drop=True)
#             split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
#         else:
#             split = None
        
#         return split


#     def return_splits(self, from_id=True, csv_path=None):

#         # import ipdb;ipdb.set_trace()
#         if from_id:
#             if len(self.train_ids) > 0:
#                 train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
#                 train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

#             else:
#                 train_split = None
            
#             if len(self.val_ids) > 0:
#                 val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
#                 val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

#             else:
#                 val_split = None
            
#             if len(self.test_ids) > 0:
#                 test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
#                 test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
            
#             else:
#                 test_split = None
            
        
#         else:
#             assert csv_path 
#             all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
#             train_split = self.get_split_from_df(all_splits, 'train')
#             val_split = self.get_split_from_df(all_splits, 'val')
#             test_split = self.get_split_from_df(all_splits, 'test')
#             # import ipdb;ipdb.set_trace()
            
#         return train_split, val_split, test_split

#     def get_list(self, ids):
#         return self.slide_data['slide_id'][ids]

#     def getlabel(self, ids):
#         return self.slide_data['label'][ids]

#     def __getitem__(self, idx):
#         return None

#     def test_split_gen(self, return_descriptor=False):

#         if return_descriptor:
#             index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
#             columns = ['train', 'val', 'test']
#             df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
#                             columns= columns)

#         count = len(self.train_ids)
#         print('\nnumber of training samples: {}'.format(count))
#         labels = self.getlabel(self.train_ids)
#         unique, counts = np.unique(labels, return_counts=True)
#         for u in range(len(unique)):
#             print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
#             if return_descriptor:
#                 df.loc[index[u], 'train'] = counts[u]
        
#         count = len(self.val_ids)
#         print('\nnumber of val samples: {}'.format(count))
#         labels = self.getlabel(self.val_ids)
#         unique, counts = np.unique(labels, return_counts=True)
#         for u in range(len(unique)):
#             print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
#             if return_descriptor:
#                 df.loc[index[u], 'val'] = counts[u]

#         count = len(self.test_ids)
#         print('\nnumber of test samples: {}'.format(count))
#         labels = self.getlabel(self.test_ids)
#         unique, counts = np.unique(labels, return_counts=True)
#         for u in range(len(unique)):
#             print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
#             if return_descriptor:
#                 df.loc[index[u], 'test'] = counts[u]

#         assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
#         assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
#         assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

#         if return_descriptor:
#             return df

    # def save_split(self, filename):
    #     train_split = self.get_list(self.train_ids)
    #     val_split = self.get_list(self.val_ids)
    #     test_split = self.get_list(self.test_ids)
    #     df_tr = pd.DataFrame({'train': train_split})
    #     df_v = pd.DataFrame({'val': val_split})
    #     df_t = pd.DataFrame({'test': test_split})
    #     df = pd.concat([df_tr, df_v, df_t], axis=1) 
    #     df.to_csv(filename, index = False)


# class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
#     def __init__(self,
#         data_dir, 
#         **kwargs):
    
#         super(Generic_MIL_Dataset, self).__init__(**kwargs)
#         self.data_dir = data_dir
#         # self.use_h5 = False
#         self.use_h5 = True

#     def load_from_h5(self, toggle):
#         self.use_h5 = toggle

#     def __getitem__(self, idx):
#         slide_id = self.slide_data['slide_id'][idx]
#         label = self.slide_data['label'][idx]
#         if type(self.data_dir) == dict:
#             source = self.slide_data['source'][idx]
#             data_dir = self.data_dir[source]
#         else:
#             data_dir = self.data_dir

#         # if not self.use_h5:
#         # 	if self.data_dir:
#         # 		full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
#         # 		features = torch.load(full_path)
#         # 		# print(features.shape)
#         # 		return features, label
            
#         # 	else:
#         # 		return slide_id, label

#         # else:
#         full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
#         with h5py.File(full_path,'r') as hdf5_file:
#             # print(hdf5_file)
#             features = hdf5_file['features'][:]
#             features2 = hdf5_file['features2'][:]
#             coords = hdf5_file['coords'][:]

#         features = torch.from_numpy(features)
#         features2 = torch.from_numpy(features2)
#         # print(slide_id)
#         # print(features.shape)

#         if hasattr(self, 'logits'):
#             return features, features2, label, self.logits[idx]
#         return features, features2, label
class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
                 csv_path,
                 **kwargs):
        super(Generic_MIL_Dataset, self).__init__(csv_path=csv_path, **kwargs)
        self.data_dir = csv_path
        self.slides = []

    def __getitem__(self, idx):
        #label = self.slide_data.iloc[idx][1]
        row = self.slide_data.iloc[idx]
        label = row.iloc[1]
        if self.args.loadonmemory:
            slide = self.slides[idx]
            patch_embeddings, region_embeddings, label = slide[0], slide[1], label
        else:
            #features = self.slide_data.iloc[idx][0].replace(" ", "")
            features = row.iloc[0]

            #data = joblib.load(features[11:])
            data = joblib.load(features)
            patch_embeddings, region_embeddings, label = data["patch"].numpy(), data["region"].numpy(), label

        return torch.Tensor(patch_embeddings), torch.Tensor(region_embeddings), label


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, args=None):
        super(Generic_Split, self).__init__(csv_path="", args=args)
        self.args = args
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.slides = []

        if args.loadonmemory:
            for idx, slide in tqdm.tqdm(enumerate(self.slide_data["slide"].values)):
                #slide = slide.replace(" ", "")
                #slide = slide[11:]
                data = joblib.load(slide)
                self.slides.append((data["patch"].numpy(), data["region"].numpy()))

    def __len__(self):
        return len(self.slide_data)


# class Generic_Split(Generic_MIL_Dataset):
#     def __init__(self, slide_data, data_dir=None, num_classes=2):
#         self.use_h5 = False
#         self.slide_data = slide_data
#         self.data_dir = data_dir
#         self.num_classes = num_classes
#         self.slide_cls_ids = [[] for i in range(self.num_classes)]
#         for i in range(self.num_classes):
#             self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

#     def __len__(self):
#         return len(self.slide_data)

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    # @property
    # def cummulative_sizes(self):
    #     warnings.warn("cummulative_sizes attribute is renamed to "
    #                   "cumulative_sizes", DeprecationWarning, stacklevel=2)
    #     return self.cumulative_sizes
        
def process_slide(slide_path):
    if not os.path.isfile(os.path.join(slide_path, "sortedv3.joblib")):
        patches = joblib.load(os.path.join(slide_path, "embeddings.joblib"))
        patch_level = patches["level"].to_numpy()
        patch_childof = patches["childof"].to_numpy()
        patch_childof[np.isnan(patch_childof)] = -1
        embeddings = patches["embedding"]
        x_coords = torch.LongTensor(patches["x"])
        y_coords=torch.LongTensor(patches["y"])
        x_coords=x_coords[patch_level==3]
        y_coords=y_coords[patch_level==3]
        size = embeddings.shape[0]
        # Get X
        x = []
        for i in range(size):
            x.append(torch.Tensor(np.matrix(embeddings[i])))
        X = torch.vstack(x)

        # Save label
        label = os.path.basename(slide_path).split("_")[-1]
        if "0" in label or "1" in label:
            label = int(label)
        else:
            if label == "tumor":
                label = 1
            else:
                label = 0
        indecesperlevel = []
        # forward input for each scale gnn
        for i in np.unique(patch_level):
            # select scale
            indeces_feats = torch.Tensor((patch_level == i).nonzero()[0]).int().view(-1)
            indecesperlevel.append(indeces_feats)
        child_index = indecesperlevel[1]
        parents_index = patch_childof[child_index]

        featshigher = X[child_index]
        featslower = X[parents_index]
        print(featshigher.size(), featslower.size())
        joblib.dump((featshigher, featslower, label,x_coords,y_coords), os.path.join(slide_path, "sortedv3.joblib"))
    else:
        featshigher, featslower, label, x_coords, y_coords= joblib.load(os.path.join(slide_path, "sortedv3.joblib"))
    return featshigher, featslower, label,x_coords, y_coords

class Split(Dataset):
    def __init__(self, data,args,name="train"):
        super(Split, self).__init__()
        self.name = name
        self.paths = data
        self.args=args
        self.slide_data=pd.DataFrame([int(os.path.basename(slide)[-1]) for slide in self.paths],columns=["labels"])
        self.data=[]
        if self.args.loadonmemory:
            for slide in self.paths:
                self.data.append(process_slide(slide))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.args.loadonmemory:
            return self.data[idx]
        else:
            return process_slide(self.paths[idx])

class CamDataset(Dataset):
    """
    Custom dataset for processing and accessing data.

    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): A function/transform that takes in an
            object and returns a transformed version. Defaults to None.
        pre_transform (callable, optional): A function/transform that takes in an
            object and returns a transformed version. Defaults to None.
        type (str, optional): Type of the dataset. Defaults to "train".
    """

    def __init__(self,source,args):
        super(CamDataset, self).__init__()
        self.source = source
        self.bags = glob.glob(os.path.join(source, "*/*"))
        self.args = args
        self.process()

    def len(self):
        return len(self.bags)

    def process(self):
        """
        Process the dataset and save processed data.
        """
        bags = glob.glob(os.path.join(self.source, "*/*"))
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        for idx, bag in enumerate(bags):
            if "test" in bag:
                if self.args.debug_mode and len(self.test_dataset)>10:
                    continue
                self.test_dataset.append(bag)
            elif "val" in bag:
                if self.args.debug_mode and len(self.val_dataset) > 10:
                    continue
                self.val_dataset.append(bag)
            else:
                if self.args.debug_mode and len(self.train_dataset) > 10:
                    continue
                self.train_dataset.append(bag)

    def __getitem__(self, idx):
        pass

    def return_splits(self):
        """
        Return the dataset split.

        Args:
            fold (int): Fold number.

        Returns:
            tuple: Train, validation and test datasets.
        """
        train_size = int(len(self.train_dataset) * 0.98)
        val_size = len(self.train_dataset) - train_size
        self.val_dataset = self.train_dataset[:val_size]
        self.train_dataset = self.train_dataset[val_size:]
        return Split(self.train_dataset,self.args,"train"), Split(self.val_dataset,self.args,"val"), Split(self.test_dataset,self.args,"test")




def print_summary(data, name):
    print("---------------------------------")
    print(f"{name} dataset summary")
    print(f"Number of slides: {len(data)}")
    classes = data["labels"].value_counts()
    name_classes = data["labels"].unique()
    print(f"Classes: {name_classes}")
    for name in name_classes:
        print(f"Number of {name} slides: {classes[name]}")

    print("---------------------------------")


class Sequential_Generic_MIL_Dataset(ContinualDataset):
    # NAME = 'seq-milv2'
    NAME = "seq-wsi"
    SETTING = 'class-il'
    # SETTING = 'task-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 4
    TRANSFORM = None

    # FOLD = 0

    def __init__(self, args):

        super(Sequential_Generic_MIL_Dataset, self).__init__(args)
        if args.cam=="only":
            self.N_TASKS=1
            self.datasets = [

                #data_path="/work/H2020DeciderFicarra/gbontempo/feats/camplit_23"
                CamDataset(self.args.data_path,args=self.args),
                #Generic_MIL_Dataset(name="lung", csv_path='lung10fold.csv',  args=self.args),
                #Generic_MIL_Dataset(name="brca", csv_path='brca10fold.csv', args=self.args),
                #Generic_MIL_Dataset(name="kidney", csv_path='kidney10fold.csv', args=self.args),
                #Generic_MIL_Dataset(name="esca", csv_path='esca10fold.csv', args=self.args)
            ]
            self.class_names = ["Breast cancer metastases"]
            self.task_names=["Breast"]

        elif args.cam=="both":
            self.N_TASKS = 5
            
            self.datasets = [
                CamDataset("/work/H2020DeciderFicarra/gbontempo/feats/camplit_23", args=self.args),
                Generic_MIL_Dataset(name="lung", csv_path='lung10fold.csv',  args=self.args),
                Generic_MIL_Dataset(name="brca", csv_path='brca10fold.csv', args=self.args),
                Generic_MIL_Dataset(name="kidney", csv_path='kidney10fold.csv', args=self.args),
                Generic_MIL_Dataset(name="esca", csv_path='esca10fold.csv', args=self.args)
            ]
            self.class_names = ["Breast normal","Breast cancer metastases","Lung Adenocarcinoma", "Lung squamous cell carcinoma", "Breast Invasive ductal",
                           "Breast Invasive lobular", "Kidney clear cell carcinoma", "Kidney papillary cell carcinoma",
                           "Esophageal adenocarcinoma", "Esophageal squamous cell carcinoma"]
            self.task_names = ["Breast", "Lung", "Breast", "Kidney", "Esca"]
        elif args.cam=="excluded":
            self.N_TASKS = 4
            self.datasets = [
                Generic_MIL_Dataset(name="lung", csv_path='lung10fold_conch.csv',  args=self.args),
                Generic_MIL_Dataset(name="brca", csv_path='brca10fold_conch.csv', args=self.args),
                Generic_MIL_Dataset(name="kidney", csv_path='kidney10fold_conch.csv', args=self.args),
                Generic_MIL_Dataset(name="esca", csv_path='esca10fold_conch.csv', args=self.args)
            ]
            self.class_names = ["Lung Adenocarcinoma", "Lung squamous cell carcinoma", 
                                "Breast Invasive ductal","Breast Invasive lobular", 
                                "Kidney clear cell carcinoma", "Kidney papillary cell carcinoma",
                           "Esophageal adenocarcinoma", "Esophageal squamous cell carcinoma"]
            self.task_names = ["Lung", "Breast", "Kidney", "Esca"]
        elif args.cam=="reverse_order":
            self.N_TASKS = 4
            self.datasets = [
                Generic_MIL_Dataset(name="esca", csv_path='esca10fold_conch.csv', args=self.args),
                Generic_MIL_Dataset(name="kidney", csv_path='kidney10fold_conch.csv', args=self.args),
                Generic_MIL_Dataset(name="brca", csv_path='brca10fold_conch.csv', args=self.args),
                Generic_MIL_Dataset(name="lung", csv_path='lung10fold_conch.csv',  args=self.args)
            ]
            self.class_names = [ "Esophageal squamous cell carcinoma","Esophageal adenocarcinoma",
                                "Kidney papillary cell carcinoma","Kidney clear cell carcinoma",
                                "Breast Invasive lobular", "Breast Invasive ductal",
                                "Lung squamous cell carcinoma", "Lung Adenocarcinoma",
                                
                          ]
            self.task_names = ["Esca","Kidney", "Breast", "Lung"]
        elif args.cam=="lung":
            self.N_TASKS = 1
            self.datasets = [
                Generic_MIL_Dataset(name="lung", csv_path='lung10fold_conch.csv',  args=self.args)
            ]
            self.class_names = ["Lung Adenocarcinoma", "Lung squamous cell carcinoma"]
            self.task_names = [ "Lung"]
        elif args.cam=="brca":
            self.N_TASKS = 1
            self.datasets = [
                Generic_MIL_Dataset(name="brca", csv_path='brca10fold_conch.csv', args=self.args)
            ]
            self.class_names = ["Breast Invasive ductal","Breast Invasive lobular"]
            self.task_names = [ "Breast"]

    def load(self):
        print("Loading data")
        self.test_loaders = []
        self.train_loaders = []
        self.val_loaders = []
        if "joint" in self.args.model:
            self.train_datasets= []
        for n in range(len(self.datasets)):
            dataset = self.datasets[n]
            train_dataset, val_dataset, test_dataset = dataset.return_splits()
            print("---------------------------------")
            print_summary(train_dataset.slide_data, "train")
            print_summary(val_dataset.slide_data, "val")
            print_summary(test_dataset.slide_data, "test")
            print("---------------------------------")
            g = torch.Generator()
            g.manual_seed(self.args.seed)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)
            if "joint" in self.args.model:
                self.train_datasets.append(train_dataset)
            else:
                self.train_loaders.append(train_loader)
            if self.args.test_on_val:
                self.test_loaders.append(val_loader)
                self.val_loaders.append(test_loader)
            else:
                self.test_loaders.append(test_loader)
                self.val_loaders.append(val_loader)
        if "joint" in self.args.model:
            train_dataset_tot = ConcatDataset(self.train_datasets)
            train_loader = DataLoader(train_dataset_tot, batch_size=1, shuffle=True)
            self.train_loaders.append(train_loader)
        print("Data loaded")



    def get_data_loaders(self, fold):
        train_loader = self.train_loaders[self.i]
        val_loader = self.val_loaders[self.i ]
        test_loader = self.test_loaders[self.i]
        self.i =(self.i+1)%(self.N_TASKS+1)
        return train_loader, val_loader, test_loader

    def get_joint_data_loaders(self, fold):
        train_loader = self.train_loaders[0]
        self.i = self.N_TASKS
        val_loader = self.val_loaders[0]
        test_loader = self.test_loaders[0]
        return train_loader, val_loader, test_loader

    @staticmethod
    def get_backbone():
        # return MNISTMLP(28 * 28, SequentialMNIST.N_TASKS
        #                 * SequentialMNIST.N_CLASSES_PER_TASK)
        # return CLAM_SB(n_classes=8)
        return HIT(num_classes=8)
        # return TransMIL(n_classes=8)

        # i_classifier = FCLayer(in_size=768, out_size=8)
        # b_classifier = BClassifier(input_size=768, output_class=8, dropout_v=0.0)
        # return MILNet(i_classifier, b_classifier)

    # def get_backbone(self,args):
    #     if args.architecture == "dsmil":
    #         model = MILNetOriginal(args)
    #         return model
    #     if args.architecture == "hit":
    #         return HIT(num_classes=8)
    #     if args.architecture == "maxpooling":
    #         return MaxPooling(args)
    #     if args.architecture == "meanpooling":
    #         return Meanpooling(args)
    #     if args.architecture == "abmil":
    #         return ABMIL(args)
    #     if args.architecture=="cocoopmil":
    #         return CoCoopMil(classnames=self.class_names,task_names=self.task_names,args=args)
    #     elif args.architecture=="conch_zeroshot":
    #         return CONCHZeroShot(classnames=self.class_names,task_names=self.task_names,args=args)
    #     elif args.architecture=="top":
    #         return build_top(args,class_names=self.class_names)
    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return f.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None


# class Sequential_Generic_MIL_Dataset(ContinualDataset):

#     NAME = 'seq-wsi'
#     SETTING = 'class-il'
#     # SETTING = 'task-il'
#     N_CLASSES_PER_TASK = 2
#     N_TASKS = 4
#     TRANSFORM = None
#     # FOLD = 0

#     datasets = [
#         Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv',
#                             data_dir= '../Dataset/TCGA-NSCLC/patch_4096/convnexts_l0l1_512_4096/',
#                             shuffle = False, 
#                             seed = 0, 
#                             print_info = True,
#                             label_dict = {'LUAD':6, 'LUSC':7},
#                             patient_strat=False,
#                             ignore=[]),
#         Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label.csv',
#                             data_dir= '../Dataset/TCGA-BRCA/patch_4096/convnexts_l0l1_512_4096/',
#                             shuffle = False, 
#                             seed = 0, 
#                             print_info = True,
#                             label_dict = {'IDC':4, 'ILC':5},
#                             patient_strat=False,
#                             ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT']),
#         Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-RCC/tcga-kidney_label.csv',
#                             data_dir= '../Dataset/TCGA-RCC/patch_4096/convnexts_l0l1_512_4096',
#                             shuffle = False, 
#                             seed = 0, 
#                             print_info = True,
#                             label_dict = {'CCRCC':2, 'PRCC':3},
#                             patient_strat=False,
#                             ignore=['CHRCC']),
#         Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-ESCA/tcga-esca_label.csv',
#                             data_dir= '../Dataset/TCGA-ESCA/patch_4096/convnexts_l0l1_512_4096/',
#                             shuffle = False, 
#                             seed = 0, 
#                             print_info = True,
#                             label_dict = {'Adenocarcinoma':0, 'Squamous cell carcinoma':1},
#                             patient_strat=False,
#                             ignore=['Tubular adenocarcinoma', 'Basaloid squamous cell carcinoma']),
#     ]
#     split_dirs = [
#         '../HIT/10fold_splits/NSCLC_100',
#         '../HIT/10fold_splits/BRCA_100',
#         '../HIT/10fold_splits/RCC_100',
#         '../HIT/10fold_splits/ESCA_100'
#     ]
#     datasets.reverse()
#     split_dirs.reverse()


#     def get_data_loaders(self, FOLD):
#         dataset = self.datasets[self.i // 2]
#         train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
#                 csv_path='{}/splits_{}.csv'.format(self.split_dirs[self.i // 2], FOLD))
#         train_loader = DataLoader(train_dataset,
#                               batch_size=1, shuffle=True, num_workers=4, collate_fn = collate_MIL)
#         val_loader = DataLoader(val_dataset,
#                               batch_size=1, shuffle=True, num_workers=4, collate_fn = collate_MIL)
#         test_loader = DataLoader(test_dataset,
#                              batch_size=1, shuffle=False, num_workers=4, collate_fn = collate_MIL)
#         # transform = transforms.ToTensor()
#         # train_dataset = MyMNIST(base_path() + 'MNIST',
#         #                         train=True, download=True, transform=transform)
#         # if self.args.validation:
#         #     train_dataset, test_dataset = get_train_val(train_dataset,
#         #                                                 transform, self.NAME)
#         # else:
#         #     test_dataset = MNIST(base_path() + 'MNIST',
#         #                         train=False, download=True, transform=transform)

#         # train, test = store_masked_loaders(train_dataset, test_dataset, self)
#         self.i += self.N_CLASSES_PER_TASK
#         self.test_loaders.append(test_loader)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         return train_loader, val_loader, test_loader
#     def get_joint_data_loaders(self, FOLD):
#         train_datasets, val_datasets, test_datasets = [], [], []
#         for n in range(self.N_TASKS):
#             dataset = self.datasets[n]
#             train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
#                     csv_path='{}/splits_{}.csv'.format(self.split_dirs[n], FOLD))
#             train_datasets.append(train_dataset)
#             val_datasets.append(val_dataset)

#             test_loader = DataLoader(test_dataset,
#                              batch_size=1, shuffle=False, num_workers=4, collate_fn = collate_MIL)
#             self.test_loaders.append(test_loader)
#             # test_datasets.append(test_dataset)
        
#         train_dataset = ConcatDataset(train_datasets)
#         val_dataset = ConcatDataset(val_datasets)
#         # test_dataset = ConcatDataset(test_datasets)

#         train_loader = DataLoader(train_dataset,
#                               batch_size=1, shuffle=True, num_workers=4, collate_fn = collate_MIL)
#         val_loader = DataLoader(val_dataset,
#                               batch_size=1, shuffle=True, num_workers=4, collate_fn = collate_MIL)
#         # test_loader = DataLoader(test_dataset,
#         #                      batch_size=1, shuffle=False, num_workers=4, collate_fn = collate_MIL)
#         # transform = transforms.ToTensor()
#         # train_dataset = MyMNIST(base_path() + 'MNIST',
#         #                         train=True, download=True, transform=transform)
#         # if self.args.validation:
#         #     train_dataset, test_dataset = get_train_val(train_dataset,
#         #                                                 transform, self.NAME)
#         # else:
#         #     test_dataset = MNIST(base_path() + 'MNIST',
#         #                         train=False, download=True, transform=transform)

#         # train, test = store_masked_loaders(train_dataset, test_dataset, self)
#         self.i = self.N_CLASSES_PER_TASK * self.N_TASKS
#         # self.test_loaders.append(test_loader)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         return train_loader, val_loader, test_loader

#     @staticmethod
#     def get_backbone():
#         # return MNISTMLP(28 * 28, SequentialMNIST.N_TASKS
#         #                 * SequentialMNIST.N_CLASSES_PER_TASK)
#         # return CLAM_SB(n_classes=8)
#         return HIT(num_classes=8)
#         # return TransMIL(n_classes=8)

#         # i_classifier = FCLayer(in_size=768, out_size=8)
#         # b_classifier = BClassifier(input_size=768, output_class=8, dropout_v=0.0)
#         # return MILNet(i_classifier, b_classifier)

#     @staticmethod
#     def get_transform():
#         return None

#     @staticmethod
#     def get_loss():
#         return F.cross_entropy

#     @staticmethod
#     def get_normalization_transform():
#         return None

#     @staticmethod
#     def get_denormalization_transform():
#         return None

#     @staticmethod
#     def get_scheduler(model, args):
#         return None