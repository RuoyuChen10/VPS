"""
Created on 2024/11/7

@author: Ruoyu Chen
Florence-2 Attribution
"""

import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import supervision as sv
from sklearn import metrics
import argparse
plt.rc('font', family="Arial")
import clip

from torchvision.ops import box_convert

from transformers import AutoProcessor, AutoModelForCausalLM


from interpretation.submodular_detection import DetectionSubModularExplanation

from tqdm import tqdm
from utils import mkdir