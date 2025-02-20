import numpy as np
from dataclasses import dataclass


COLORS_DICT = dict(
    TITANWEISS=[237, 238, 240],     # 1
    ZITRONENGELB=[244, 207, 57],    # 2
    GELBOCKER=[178, 117, 50],       # 3
    ORANGEGELB=[255, 86, 21],       # 4
    ZINNOBERROT=[254, 59, 27],      # 5
    SCHARLACHROT=[215, 33, 29],     # 5
    PURPURROT=[178, 25, 30],        # 6
    SMARAGDGRUEN=[53, 165, 83],     # 7
    VIRIDIANGRUEN=[0, 117, 85],     # 8
    SAFTGRUEN=[49, 89, 37],         # 9
    HIMMELBLAU=[39, 132, 191],      # 10
    CYANBLAU_DUNKEL=[3, 77, 150],   # 11
    PHTHALOCYANINBLAU=[19, 17, 82], # 12
    SIENNA=[164, 96, 47],           # 13
    SIENNA_GEBRANNT=[179, 73, 34],  # 14
    UMBRA_GEBRANNT=[101, 57, 44],   # 15
    VANDYKE_BRAUN=[38, 37, 42],     # 16
    SCHWARZ=[20, 12, 25])           # 17

COLORS = np.array(list(COLORS_DICT.values()))

