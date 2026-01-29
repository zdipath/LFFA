# Foundation 
import os
from pathlib import Path
def get_augmentation_path(study, fm,args):
    ex_fm_dict = {'chief': 'chief',
                  'feather': 'conch_v1_5',
                  'titan': 'conch_v1_5',
                  'gigapath': 'gigapath',
                  'prism': 'virchow',
                  'tangle': 'uni_v1'}
    ex_fm = ex_fm_dict[fm]
    feature_path = ''
    result_path = os.path.join(result_path,args.aug_method)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return feature_path, result_path



