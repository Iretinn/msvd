import torch.nn as nn
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


def get_backbone(args):
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    return encoder, config

