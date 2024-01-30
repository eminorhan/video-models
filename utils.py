"""
Util functions
"""
import torch
import models_mae, models_vit
from huggingface_hub import hf_hub_download

def get_available_models():
    available_models = [
        'mae_say_none', 'mae_s_none', 'mae_kinetics_none', 'mae_kinetics-200h_none',
        'vit_say_none', 'vit_s_none', 'vit_kinetics_none', 'vit_kinetics-200h_none',
        'vit_say_ssv2-50shot', 'vit_s_ssv2-50shot', 'vit_kinetics_ssv2-50shot', 'vit_kinetics-200h_ssv2-50shot',
        'vit_say_kinetics-50shot', 'vit_s_kinetics-50shot', 'vit_kinetics_kinetics-50shot', 'vit_kinetics-200h_kinetics-50shot',
        ]

    return available_models

def load_model(model_name):

    # parse identifier
    model_type, pretrain_data, finetune_data = model_name.split('_')

    # checks
    assert model_type in ['mae', 'vit']
    assert pretrain_data in ['say', 's', 'kinetics', 'kinetics-200h'], 'Unrecognized pretraining data!'
    assert finetune_data in ['none', 'ssv2-50shot', 'kinetics-50shot'], 'Unrecognized finetuning data!'

    # download checkpoint from hf
    ckpt_filename = pretrain_data + '_' + finetune_data + '.pth'
    ckpt = hf_hub_download(repo_id='eminorhan/video-models', filename=ckpt_filename)

    if model_type.startswith('mae'):
        model = models_mae.mae_vit_huge_patch14()
        ckpt = torch.load(ckpt, map_location='cpu')
        msg = model.load_state_dict(ckpt['model'], strict=True)
        print(f'Loaded with message: {msg}')
    elif model_type.startswith('vit'):
        if finetune_data.startswith('ssv2'):
            num_classes = 174
        elif finetune_data.startswith('kinetics'):
            num_classes = 700
        else:
            num_classes = None
        model = models_vit.vit_huge_patch14(num_classes=num_classes)
        ckpt = torch.load(ckpt, map_location='cpu')['model']
        msg = model.load_state_dict(ckpt, strict=False)
        print(f'Loaded with message: {msg}')

    return model

