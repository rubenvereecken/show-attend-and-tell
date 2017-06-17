from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', dest='data_path', default='./data')
    parser.add_argument('--image-path', dest='image_path', default='./image')
    parser.add_argument('--model-path', dest='model_path', default='model/lstm')
    parser.add_argument('--test-model', dest='test_model', default='model/lstm/model-10')

    args = parser.parse_args()

    # load train dataset
    data = load_coco_data(data_path=args.data_path, split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path=args.data_path, split='val')

    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=128, update_rule='adam',
                                          learning_rate=0.001, print_every=1000, save_every=1,
                                          image_path=args.image_path + '/', # Why the trailing slash?
                                    pretrained_model=None, model_path=args.model_path, test_model=args.test_model,
                                     print_bleu=True, log_path='log/')

    solver.train()

if __name__ == "__main__":
    main()
