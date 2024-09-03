from transformers import RobertaConfig, RobertaModel

def get_backbone(args):
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    return encoder, config

