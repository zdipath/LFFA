import sys
import os

import torch
from models.CHIEF import CHIEF
from models.gigapath.slide_encoder import create_model
from src.builder import create_model_feather

from models.mmssl import MMSSL
from transformers import AutoModel
from utils.process_args import process_args
from collections import OrderedDict
import os
import json


def set_args(args, config_from_model):
    exp_code = os.path.split(os.path.normpath(args['pretrained']))[-1]
    args['study'] = exp_code.split('_')[0]
    for key in ['wsi_encoder', 'activation', 'method', 'n_heads', 'hidden_dim', 'rna_encoder', 'embedding_dim', 'rna_token_dim']:
        args[key] = config_from_model[key]

    args["rna_reconstruction"] = True if args["method"] == 'tanglerec' else False 
    args["intra_modality_wsi"] = True if args["method"] == 'intra' else False 
    return args 

def read_config(path_to_config):
    with open(os.path.join(path_to_config, 'config.json')) as json_file:
        data = json.load(json_file)
        return data 
     
def restore_model(model, state_dict):
    
    sd = list(state_dict.keys())
    contains_module = any('module' in entry for entry in sd)
    
    if not contains_module:
        model.load_state_dict(state_dict, strict=True)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)

    return model


def return_model(fm):
    print('loading model...',end=' ')
    if fm == "chief":
        model = CHIEF(size_arg="small", dropout=True, n_classes=2)
        td = torch.load(r'models_weights/chief/CHIEF_pretraining.pth',map_location="cpu")
        model.load_state_dict(td, strict=True)

    elif fm == "feather":
        model = create_model_feather('abmil.base.conch_v15.pc108-24k', num_classes=5)

    elif fm == "titan":

        model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)

    elif fm == "gigapath":
        model = create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            "gigapath_slide_enc12l768d",
            1536,
            global_pool=True,

        )

    elif fm == "prism":

        model = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)

    elif fm == "tangle":
        args = process_args()
        args['pretrained'] = "models_weights/tanglev2_mhabmil/"
        assert args['pretrained'] is not None, \
            "Must provide a path to a pretrained dir."
        config_from_model = read_config(args['pretrained'])
        args = set_args(args, config_from_model)

        model = MMSSL(
            config=args,
            n_tokens_rna=int(args["rna_token_dim"]),
        )


    model.eval()
    print('done!')
    return model


if __name__ == "__main__":
    fms = ["chief", "feather", "titan", "gigapath", "prism", "tangle"]
    for fm in fms:
        model = return_model(fm)
        print(fm)
        print(model)