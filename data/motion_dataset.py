import numpy as np
import os
import random as rd
from glob import glob
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# directories containing text descriptions and motion files
text_dir = './data/texts/'  # directory containing all .txt description files
motion_dir = './data/motions/'  # directory containing all .npy motion files

# get sorted list of all text files and motion files
all_text = sorted(glob(os.path.join(text_dir, '*.txt')))
all_motion = sorted(glob(os.path.join(motion_dir, '*.npy')))


class MotionDataset(Dataset):
    def __init__(self,
                 file: str = "train",
                 tokenizer_name: str = 't5-small'):
        """
        Params
        -------
        file : str
            File containing the list of data samples to be used.
            "val", "train" or "test". Defaults to "train".
        """

        assert file in ["train", "val", "test"], "file argument must be one of 'train', 'val', or 'test'"

        # path of the text descripions and motions directory
        # .../text/
        # .../motions/
        self.text_dir = text_dir
        self.motion_dir = motion_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        # get sorted list of all text files and motion files
        self.all_text = all_text
        self.all_motion = all_motion

        self.files_name = []
        with open('./data/' + file + '.txt', 'r') as f:
            self.files_name = f.read().splitlines()

        self.text_files = sorted([i for i in self.all_text if os.path.basename(i).split('.')[0] in self.files_name])
        self.motion_files = sorted([i for i in self.all_motion if os.path.basename(i).split('.')[0] in self.files_name])
        
        self.motion_frames = [
            np.load(m).shape[0] for m in self.motion_files
        ]

    def __len__(self):
        return len(self.files_name)
    
    def __getitem__(self, idx):
        print(idx)
        # read npy motion file
        motion = np.load(self.motion_files[idx])

        # get the corresponding description for the associated motion
        with open(self.text_files[idx]) as f:
            descriptions = [
                caption.split('#')[0].capitalize() for caption in f.readlines()
            ]
        
        #TODO: deal with multiple captions
        text = rd.choice(descriptions)

        # Tokenize the text description
        tokens = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

        return {
            "motion": motion,
            "captions": text,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }