import tqh
import numpy as np
import sys

log = sys.stdout

# 4 tiles of data from remotesensingdata.gov.scot
# DSM Phase 5 - Crown copyright Scottish Government and Fugro (2020)
north_queensferry = tqh.load_tiles([['NT18SW_50CM_DSM_PHASE5.tif', 'NT18SE_50CM_DSM_PHASE5.tif'],
                                    ['NT17NW_50CM_DSM_PHASE5.tif', 'NT17NE_50CM_DSM_PHASE5.tif']],
                                   min_val=0.0, base_dir='.', subsample_lidar = 1, log=log)

tqh.perspective_shadow_lit(north_queensferry, upsample = 2, focus=(6000,5000),
                           light_clr=(3,)*3, # White
                           sky_clr=(0.8,)*3, # Ambient light
                           pix_shape=(1920,1280), # image shape
                           light_dir = (-0.4,-0.7,0.2), camera_angles=(2.35*np.pi, -0.7, 32,0.4),
                           save_name='example_raytrace.png',
                           log=log) # Log messages





