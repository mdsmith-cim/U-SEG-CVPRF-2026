from timm import create_model
import argparse
from pprint import pprint


def get_parser():
    parser = argparse.ArgumentParser(description='Prints information about a specified timm (pytorch-image-models) model')
    parser.add_argument('model', help='timm model name')
    parser.add_argument('--pretrained', help='If specified, load pretrained model', action='store_true')
    parser.add_argument('--features_only', help='If specified, omits head (e.g. classifier) from network', action='store_true')
    parser.add_argument('--print-model-structure', help='Prints entire model structure. May be very big.', action='store_true')
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    model = create_model(args.model, pretrained=args.pretrained, features_only=args.features_only)
    print(f'For model {args.model} possible outputs are:')
    pprint(model.feature_info.get_dicts())
    if args.print_model_structure:
        pprint(model)