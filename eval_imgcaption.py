#-------------------------------------------------------------
# File     : eval_imagecaption.py
# Author   : Yueting Wu
# Date     : 230918
# Function : evauate image-caption model
#-------------------------------------------------------------
from aac_metrics import evaluate
import torch
import numpy as np
import torch.nn.functional as nnf
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
# from google.colab import files
import PIL.Image
from IPython.display import Image
import models_tl
import models_vit
from image_text_dl import TextImageDataset
import argparse
import pickle

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=5, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy() #[beam_size, entry_length]
    
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts[0]


def llm_generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=5, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.lm.transformer.word_embeddings(tokens)
        for i in range(entry_length):
            
            outputs = model.lm(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.lm.transformer.word_embeddings(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy() #[beam_size, entry_length]
    
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts[0]


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=20,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def prepare_model(chkpt_dir, arch='Caption_MLP_base'):
    # build model
    model = getattr(models_tl, arch)()
    # final_model = getattr(models_tl, arch)()
    # load model
    if chkpt_dir != None:
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)

        # model_state_dict = model.state_dict()
        # final_model_state_dict = final_model.state_dict()
        
        # for name, param in model_state_dict.items():
        #     if "project" in name:
        #         final_model_state_dict[name] = param
        # final_model.load_state_dict(final_model_state_dict)
    return model

def prepare_model_vit(chkpt_dir, arch='vit_large_patch16'):
    # build model
    model = models_vit.__dict__['vit_large_patch16'](num_classes=2)
    # load model
    if chkpt_dir != None:
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

def get_args_parser():
    parser = argparse.ArgumentParser('evaluate image caption', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    #############################################################################################################################
    # need modification
    #############################################################################################################################
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL', 
                        help='Name of model to train')
    
    parser.add_argument('--mode', default='lcond', type=str, 
                        choices=['lcond','lgen'],help='mode of chkpt model')
    parser.add_argument('--data_path', default='/export/home/wuyueting/thyroid_data/CLIPdata_folder/CLIPdata_image_cut_BM_output/', type=str,
                        help='dataset path')
    parser.add_argument('--prefix_length', default=10, type=int,
                        help='prefix length')
    parser.add_argument('--use_beam_search', default=True, type=bool,
                        help='if True, generate beam search, if False, use generate2')
    

    return parser

def precise_acc(text:str, pred:str):
    features = text.split(", ")
    pred_features = pred.split(",")
    pred_features = [x.strip() for x in pred_features]
    # print(features,pred_features)
    # Jaccard相似度，通过计算两个集合的交集大小与并集大小的比较来衡量相似性

    intersection = len(set(features).intersection(pred_features))
    union = len(set(features).union(pred_features))

    # print(intersection,union,set(features).union(pred_features))
    jaccard_similarity = intersection/union

    return jaccard_similarity

def raw_acc(text:str, pred:str):
    features = text.split(", ")
    
    score=0

    for feature in features:
        if feature in pred:
            score +=1
    
    return score/len(features)

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM

def main(args):
    # load language model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("/export/home/wuyueting/ReferenceModel/LLM/falcon/falcon-7b")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model_path = "/export/home/wuyueting/Encoder/mae_like/mae_main_wyt_revised/checkpoints/Prefix_LLM_MLP_base_new/feature_unshuffle/thyroid_prefix-009-project.pt"
    print("Loading language model...")
    model = prepare_model(model_path,'Prefix_LLM_MLP_base')
    model = model.eval() 
    model = model.to(device)
    print("Language model is loaded!")

    # load image model
    # print("Loading image model")
    # model_vit = prepare_model_vit("/export/home/wuyueting/Encoder/mae_like/mae_main_wyt_revised/output_dir/large/i_lcond/finetune/0.85/large_lcond_finetune_85_bestauc.pth")
    # model_vit = model_vit.eval() 
    # model_vit = model_vit.to(device)
    # print("Image model is loaded!")
#
    # transform_val = transforms.Compose([
    #         transforms.Resize(224),  # 3 is bicubic
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.22, 0.22, 0.22], std=[0.08, 0.08, 0.08])])
    
    # dataset_train =  TextImageDataset(args.data_path, "test",text_mode="feature_unshuffle", transform = transform_val, custom_tokenizer="rawtext")
    testdata_path = "data/large_lcond_finetune_feature_unshuffle_test.pkl"
    with open(testdata_path, "rb") as f:
        dataset_test = pickle.load(f) 
    ites = dataset_test["img_embeddings"]
    texts = dataset_test["texts"]
    data_length = len(ites)
    candidates = []
    references = []
    acc_score = []
    if args.use_beam_search:
        generatefunction = llm_generate_beam
    else:
        generatefunction = generate2
    
    result_path = model_path.replace(".pt",".txt")
    f = open(result_path,"w")
    for i in tqdm(range(data_length)):
        with torch.no_grad():
            # ite = model_vit.encode_image(img_tensor.unsqueeze(0).to(device))
            ite = ites[i].to(device)
            prefix_embed = model.project(ite).reshape(1, args.prefix_length, -1)
        generated_text_prefix = generatefunction(model, tokenizer, embed=prefix_embed,entry_length=30)
        
        text = texts[i]
        f.write("number:"+str(i)+"\n")
        f.write("Text:"+text+"\n")
        f.write("generate:"+generated_text_prefix+"\n")
        references.append([text])
        candidates.append(generated_text_prefix)
        acc_score.append(raw_acc(text, generated_text_prefix))

    average = sum(acc_score) / len(acc_score)
    print(average)
    print("Start calculating scores...")
    corpus_scores, _ = evaluate(candidates, references)
    print(corpus_scores)

   
    f.write(str(average)+"\n")
    f.write(str(corpus_scores))
    f.close()


if __name__ =="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)