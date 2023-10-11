#-------------------------------------------------------------
# File     : main_imagecaption.py
# Author   : Yueting Wu
# Date     : 230907
# Function : tranform image embedding into language
# Reference:
# ClipCap: CLIP Prefix for Image Captioning
#-------------------------------------------------------------
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from image_text_dl import TextImageTokenDataset
import models_tl
import argparse
import os
from transformers import AdamW, get_linear_schedule_with_warmup
import sys
from tqdm import tqdm

def train(dataset: TextImageTokenDataset,  args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "",prefix:bool=True):

    model = models_tl.__dict__[args.model]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    print(batch_size)
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (img, lang, lang_mask) in enumerate(train_dataloader):
            model.zero_grad()
            img, lang, lang_mask = img.to(device, dtype=torch.float32), lang.to(device), lang_mask.to(device)
            loss = model(img, lang, lang_mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if prefix:
            if epoch % args.save_every == 0 or epoch == epochs - 1:
                layer_name_keyword = "project"
                layers_to_save = [name for name, _ in model.named_parameters() if layer_name_keyword in name]

                layer_params = {}
                for layer_name in layers_to_save:
                    layer_params[layer_name] = model.state_dict()[layer_name]
                torch.save(
                    layer_params,
                    os.path.join(output_dir, f"{output_prefix}-{epoch:03d}-project.pt"),
                )
        else:
            if epoch % args.save_every == 0 or epoch == epochs - 1:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
                )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./checkpoints/feature_unshuffle/')
    parser.add_argument('--prefix', default='thyroid_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--text_mode', default='feature_unshuffle', type=str,
                        help='text_mode', choices=["feature","sentence","feature_unshuffle"])
    parser.add_argument('--model', default="Prefix_LLM_MLP_base", type=str, metavar='MODEL')
    parser.add_argument('--lm', default="falcon-7b", type=str)
    args = parser.parse_args()
    
    if args.lm == "falcon-7b":
        assert "LLM"in args.model, "Image Caption model is not suitable for LLM"

    data = TextImageTokenDataset(data_path = f"data/large_lcond_finetune_{args.lm}_{args.text_mode}_train.pkl")

    output_dir = './checkpoints/'+args.model+"_newbs32/"+args.text_mode+"/"
    train(data, args, output_dir=output_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()