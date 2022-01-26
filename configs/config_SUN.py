MAIN_DATAPATH   = 'data/SUNP/'
VAL_DATAPATH    = 'data/SUNP/validation/'
TEST_DATAPATH   = 'data/SUNP/test/'

sun_model_paths = {}
sun_model_paths["vanilla"] = 'model/ale_sun.pt'
sun_model_paths["smooth"] = 'model/ale_sun_smoothed.pt'
sun_model_paths["ant"] = "model/sun_ant.pt"
sun_model_paths["augmix"] = 'model/sun_augmix.pt'