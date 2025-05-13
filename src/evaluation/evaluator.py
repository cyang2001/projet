# -*- coding: utf-8 -*-
"""
Created on Sun May  4 11:49:42 2025

@author: fross
"""

import numpy as np
import scipy.io
import logging
from typing import Dict, Tuple, Any

from utils.utils import get_logger

def evaluation(BDREF_path, BDTEST_path, resize_factor):
    logger = get_logger(__name__)
    
    try:
        BDREF = scipy.io.loadmat(BDREF_path)['BD']
        BDTEST = scipy.io.loadmat(BDTEST_path)['BD']
    except Exception as e:
        logger.error(f"Failed to load .mat file: {e}")
        return None
    
    BDTEST[:, 1:5] = resize_factor * BDTEST[:, 1:5]
    
    I = np.mean(BDREF[:, 1:3], axis=1)
    J = np.mean(BDREF[:, 3:5], axis=1)
    D = np.round((BDREF[:, 2] - BDREF[:, 1] + BDREF[:, 4] - BDREF[:, 3]) / 2)
    maxDecPer = 0.1
    
    confusionMatrix = np.zeros((14, 14), dtype=int)
    plusVector = np.zeros(14, dtype=int)   
    minusVector = np.zeros(14, dtype=int)   
    processed = np.zeros(BDREF.shape[0], dtype=bool)
    
    for k in range(BDTEST.shape[0]):
        n = BDTEST[k, 0] 
        ind = np.where(BDREF[:, 0] == n)[0]  
        
        i = np.mean(BDTEST[k, 1:3])
        j = np.mean(BDTEST[k, 3:5])
        
        d = np.sqrt((I[ind] - i) ** 2 + (J[ind] - j) ** 2)
        
        if len(d) > 0:
            mind = np.min(d)
            p = np.argmin(d)
            kref = ind[p]
            
            if mind <= maxDecPer * D[kref]:
                confusionMatrix[int(BDREF[kref, 5]) - 1, int(BDTEST[k, 5]) - 1] += 1
                processed[kref] = True
            else:
                plusVector[int(BDTEST[k, 5]) - 1] += 1
    

    for k in np.where(~processed)[0]:
        minusVector[int(BDREF[k, 5]) - 1] += 1
    

    logger.info("\n\n---------------\nConfusion matrix....\n")
    for k in range(14):
        row = ' '.join(f"{val:3d}" for val in confusionMatrix[k])
        total = np.sum(confusionMatrix[k])
        logger.info(f"{row}  : {total:3d} : + {plusVector[k]:3d} : - {minusVector[k]:3d} :")
    
    logger.info("... ... ... ... ... ... ... ... ... ... ... ... ... ...")
    col_totals = [np.sum(confusionMatrix[:, k]) for k in range(14)]
    logger.info(' '.join(f"{val:3d}" for val in col_totals))
    

    logger.info("\n\n---------------\nRecognition rate")
    reconnus = 0  
    recognition_rates = []
    
    for k in range(14):
        total = np.sum(confusionMatrix[k]) + minusVector[k]
        taux_reconnu = 100 * confusionMatrix[k, k] / total if total > 0 else 0
        taux_plus = 100 * plusVector[k] / total if total > 0 else 0
        logger.info(f"{k+1:2d} : {taux_reconnu:5.2f} %  - false positive rate: {taux_plus:5.2f} %")
        reconnus += confusionMatrix[k, k]
        recognition_rates.append(taux_reconnu)
    
    total_all = np.sum(confusionMatrix) + np.sum(minusVector)
    global_recognition_rate = 100 * reconnus / total_all
    false_positive_rate = 100 * np.sum(plusVector) / total_all
    
    logger.info("---------------")
    logger.info(f"Global recognition rate = {global_recognition_rate:.2f} %")
    logger.info(f"False positive rate = {false_positive_rate:.2f} %")
    logger.info("---------------")
    
    stats = {
        "confusion_matrix": confusionMatrix,
        "false_positives": plusVector,
        "false_negatives": minusVector,
        "recognition_rates": recognition_rates,
        "global_recognition_rate": global_recognition_rate,
        "false_positive_rate": false_positive_rate
    }
    
    return stats

