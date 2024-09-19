from src.config.default import _CN as cfg

cfg.MOMAMATCHER.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.MOMAMATCHER.MATCH_COARSE.SPARSE_SPVS = False

cfg.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875 
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_GAMMA = 0.5
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5 

cfg.MOMAMATCHER.RESOLUTION = (32, 8, 2)  # (16,8,2)
cfg.DATASET.MGDPT_DF = cfg.MOMAMATCHER.RESOLUTION[0]
cfg.MOMAMATCHER.MATCH_COARSE.T_K = -1 

cfg.TRAINER.OPTIMIZER = 'adamw'
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.MOMAMATCHER.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
cfg.MOMAMATCHER.COARSE.COARSEST_LEVEL= [20, 20] #[26, 26]
