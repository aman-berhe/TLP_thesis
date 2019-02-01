# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:49:33 2018

@author: berhe
"""

import ExtractingFrameFeatures as eff
import numpy as np

videoFile='/people/berhe/Bureau/video/GameOfThrones.Season01.Episode01.mkv'

framesFeatures,timeStamp,frameIds=eff.getframes(1,videoFile)
framesFeatures=np.array(framesFeatures)
np.save('framesFeaturesGPU',framesFeatures)