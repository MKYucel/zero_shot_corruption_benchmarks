MAIN_DATAPATH   = 'data/CUBP/'
VAL_DATAPATH    = 'data/CUBP/validation/'
TEST_DATAPATH   = 'data/CUBP/test/'

cub_model_paths = {}
cub_model_paths["vanilla"] = 'model/ale_cub.pt'
cub_model_paths["smooth"] = 'model/ale_cub_smoothed.pt'
cub_model_paths["ant"] = 'model/cub_ant.pt'
cub_model_paths["augmix"] = 'model/cub_augmix.pt'