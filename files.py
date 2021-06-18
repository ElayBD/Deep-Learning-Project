# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 18:56:24 2021

@author: PC
"""
import os
import random
from PIL import Image 

def transferImages(ogpath,pathtrain,pathtest):
    """
    Parameters
    ----------
    ogpath : string
        DESCRIPTION: the path to the dataset in the original folder 
    pathtrain : string
        DESCRIPTION: the path to the train folder
    pathtest : string
        DESCRIPTION: the path to the test folder

    Returns
    -------
    None.

    """
    files_list = os.listdir(ogpath)#creating the list of the images
    for i in range(int(len(files_list)*0.8)):
        file=random.choice(files_list)
        os.rename(os.path.join(ogpath,file),os.path.join(pathtrain,file))
        files_list.remove(file)
    for i in range(len(files_list)):
        file=random.choice(files_list)
        os.rename(os.path.join(ogpath,file),os.path.join(pathtest,file))
        files_list.remove(file)
            
def transferimagesBackToOG(currentpath,destenationpath):
    """
    
    Parameters
    ----------
    currentpath : string 
        DESCRIPTION: the path to the dataset in the selected folder
    destenationpath : string
        DESCRIPTION:the path to the original folder of the dataset

    Returns
    -------
    None.

    """
    files_list = os.listdir(currentpath)
    for file in files_list:
        os.rename(os.path.join(currentpath,file),os.path.join(destenationpath,file))


def resize(path):
    """
    
    Parameters
    ----------
    path : string
        DESCRIPTION: the path to the files that will resize.

    Returns
    -------
    None.

    """
    filelist= os.listdir(path)
    for file in filelist:
        File = Image.open(os.path.join(path, file))
        File= File.resize((128,128))
        File.save(os.path.join(path, file))
