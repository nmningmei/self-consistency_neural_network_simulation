#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:00:57 2022

@author: nmei
"""

from utils_deep import dataloader



if __name__ == "__main__":
    
    dataloader_train = dataloader(dataset_name = 'CIFAR10',download = True,train = True)




