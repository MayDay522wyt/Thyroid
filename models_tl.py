#-------------------------------------------------------------
# File     : models_tl.py
# Author   : Yueting Wu
# Date     : 230907
# Function : tranform image embedding into language
# Reference:
# ClipCap: CLIP Prefix for Image Captioning
#-------------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
from einops import rearrange, repeat    

from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from image_text_dl import TextImageTokenDataset

   
class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=F.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerMapper(nn.Module):
    def __init__(self, dim_vit: int, dim_embedding: int, prefix_length: int, clip_length: int,
                  depth: int = 8, num_heads =8, mlp_ratio = 4.,norm_layer=nn.LayerNorm):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = nn.ModuleList([
            Block(dim_embedding, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.linear = nn.Linear(dim_vit, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
    
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        for blk in self.transformer:
            prefix = blk(prefix)
        out = prefix[:, self.clip_length:]
        return out

class ImageCaptionModel(nn.Module):
    """
    use MAE embedding generates image caption
    """
    def __init__(self, prefix_length: int = 10, clip_length: Optional[int] = None,
                  prefix_size: int = 512, # 如果是vit-large 就是1024,
                mapping_type: MappingType = MappingType.MLP):
        super(ImageCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt.resize_token_embeddings(len(tokenizer))
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1] #768
        if mapping_type == MappingType.MLP:
            self.project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            clip_length = 10
            self.project = TransformerMapper(dim_vit=prefix_size, dim_embedding=self.gpt_embedding_size, prefix_length=prefix_length,
                                                                     clip_length=clip_length)
            
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward_gpt(self, prefix: torch.Tensor, tokens: torch.Tensor,  mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        tokens: [bs, 40]
        prefix: [bs, 1024]
      
        .  mask: [bs, 40]
        embedding_text: [bs, 40, 768]
        prefix_project: [bs, 10, 768]
        mask: [bs, 40+10]
        """
        # print(tokens)
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.project(prefix) # [bs, 7680]
        prefix_projections = prefix_projections.view(-1, self.prefix_length, self.gpt_embedding_size) #[bs, 10, 768]
        img_mask = torch.ones_like(prefix_projections[..., -1], dtype=mask.dtype)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        mask_cat = torch.cat([img_mask,mask],dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask_cat)
        
        return out

    def forward_loss(self, outputs, tokens):
        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        print(logits,tokens)
        return loss

    def forward(self, prefix: torch.Tensor, tokens: torch.Tensor,  mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        out = self.forward_gpt(prefix, tokens, mask)
        loss = self.forward_loss(out,tokens)
        return loss

class ImageCaptionModel_LLM(nn.Module):
    """
    use MAE embedding generates image caption
    """
    def __init__(self, lm: str = "/export/home/wuyueting/ReferenceModel/LLM/falcon/falcon-7b",prefix_length: int = 10, clip_length: Optional[int] = None,
                  prefix_size: int = 512, # 如果是vit-large 就是1024
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ImageCaptionModel_LLM, self).__init__()
        self.prefix_length = prefix_length

        tokenizer = AutoTokenizer.from_pretrained(lm)
        
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.lm = AutoModelForCausalLM.from_pretrained(lm)
        self.lm.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        
        self.lm_embedding_size = self.lm.transformer.word_embeddings.weight.shape[1] # 对于falcon-7b-isntruct好像是一个四位数
        if mapping_type == MappingType.MLP:
            self.project = MLP((prefix_size, (self.lm_embedding_size * prefix_length) // 2,
                                     self.lm_embedding_size * prefix_length))
            # print(self.project)
        else:
            clip_length = 10
            self.project = TransformerMapper(prefix_size, self.lm_embedding_size, prefix_length,
                                                                     clip_length, num_layers)
            
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward_lm(self, prefix: torch.Tensor, tokens: torch.Tensor,  mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        tokens: [bs, 40]
        prefix: [bs, 1024]
      
        .  mask: [bs, 40]
        embedding_text: [bs, 40, 768]
        prefix_project: [bs, 10, 768]
        mask: [bs, 40+10]
        """
        # print(tokens)
        embedding_text = self.lm.transformer.word_embeddings(tokens)
        prefix_projections = self.project(prefix) # [bs, 7680]
        prefix_projections = prefix_projections.view(-1, self.prefix_length, self.lm_embedding_size) #[bs, 10, 768]
        img_mask = torch.ones_like(prefix_projections[..., -1], dtype=mask.dtype)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        mask_cat = torch.cat([img_mask,mask],dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.lm(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask_cat)
        
        return out

    def forward_loss(self, outputs, tokens):
        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        return loss

    def forward(self, prefix: torch.Tensor, tokens: torch.Tensor,  mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        out = self.forward_lm(prefix, tokens, mask)
        loss = self.forward_loss(out,tokens)
        return loss

class ImageCaptionPrefix_LLM(ImageCaptionModel_LLM):
    """
    use MAE embedding generates image caption
    """
    def __init__(self, lm: str = "/export/home/wuyueting/ReferenceModel/LLM/falcon/falcon-7b",prefix_length: int = 10, clip_length: Optional[int] = None,
                  prefix_size: int = 512, # 如果是vit-large 就是1024
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ImageCaptionModel_LLM, self).__init__()
        self.prefix_length = prefix_length

        tokenizer = AutoTokenizer.from_pretrained(lm)
        
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.lm = AutoModelForCausalLM.from_pretrained(lm)
        self.lm.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        for name, param in self.lm.named_parameters():
            param.requires_grad = False
                
        self.lm_embedding_size = self.lm.transformer.word_embeddings.weight.shape[1] # 对于falcon-7b-isntruct好像是一个四位数
        if mapping_type == MappingType.MLP:
            self.project = MLP((prefix_size, (self.lm_embedding_size * prefix_length) // 2,
                                     self.lm_embedding_size * prefix_length))
            # print(self.project)
        else:
            clip_length = 10
            self.project = TransformerMapper(prefix_size, self.lm_embedding_size, prefix_length,
                                                                     clip_length, num_layers)
            
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward_lm(self, prefix: torch.Tensor, tokens: torch.Tensor,  mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        tokens: [bs, 40]
        prefix: [bs, 1024]
      
        .  mask: [bs, 40]
        embedding_text: [bs, 40, 768]
        prefix_project: [bs, 10, 768]
        mask: [bs, 40+10]
        """
        # print(tokens)
        embedding_text = self.lm.transformer.word_embeddings(tokens)
        prefix_projections = self.project(prefix) # [bs, 7680]
        prefix_projections = prefix_projections.view(-1, self.prefix_length, self.lm_embedding_size) #[bs, 10, 768]
        img_mask = torch.ones_like(prefix_projections[..., -1], dtype=mask.dtype)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        mask_cat = torch.cat([img_mask,mask],dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.lm(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask_cat)
        
        return out

    def forward_loss(self, outputs, tokens):
        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        return loss

    def forward(self, prefix: torch.Tensor, tokens: torch.Tensor,  mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        out = self.forward_lm(prefix, tokens, mask)
        loss = self.forward_loss(out,tokens)
        return loss
    # def parameters(self, recurse: bool = True):
    #     return self.project.parameters()

    # def train(self, mode: bool = True):
    #     super(ImageCaptionPrefix_LLM, self).train(mode)
    #     self.lm.eval()
    #     print("set language model eval")
    #     return self

def caption_mlp_base_1024(**kwargs):
    model = ImageCaptionModel(prefix_length= 10,prefix_size = 1024, # 如果是vit-large 就是1024
                                     mapping_type=MappingType.MLP, **kwargs)
    return model

def caption_LLM_mlp_base_1024(**kwargs):
    model = ImageCaptionModel_LLM(lm= "/export/home/wuyueting/ReferenceModel/LLM/falcon/falcon-7b",
                                  prefix_length= 10,
                                  prefix_size = 1024, # 如果是vit-large 就是1024
                                  mapping_type=MappingType.MLP, **kwargs)
    return model

def prefix_LLM_mlp_base_1024(**kwargs):
    model = ImageCaptionPrefix_LLM(lm= "/export/home/wuyueting/ReferenceModel/LLM/falcon/falcon-7b",
                                  prefix_length= 10,
                                  prefix_size = 1024, # 如果是vit-large 就是1024
                                  mapping_type=MappingType.MLP, **kwargs)
    return model

def prefix_LLM_transformer_base_1024(**kwargs):
    model = ImageCaptionPrefix_LLM(lm= "/export/home/wuyueting/ReferenceModel/LLM/falcon/falcon-7b",
                                  prefix_length= 10,
                                  prefix_size = 1024, # 如果是vit-large 就是1024
                                  mapping_type=MappingType.Transformer, **kwargs)
    return model
Caption_MLP_base = caption_mlp_base_1024
Caption_LLM_MLP_base = caption_LLM_mlp_base_1024
Prefix_LLM_MLP_base = prefix_LLM_mlp_base_1024
Prefix_LLM_transformer_base = prefix_LLM_transformer_base_1024


if __name__ =="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = Prefix_LLM_MLP_base()
    # model.to(device)
    """
        test
        """
    
    print(model)
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model.lm,
    #     tokenizer=model.tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     device_map="auto",
    # )

    # sequences = pipeline(
    # "We are analyzing the malignancy of nodule according to the generated report. Here are a few examples:\n\
    # taller-than-wide nodule, solid composition, hypoechoic uneven echo, irregular shape, uncalcified, unclear margin->malignant\n\
    # wider-than-tall nodule, solid composition, clear margin, irregular shape, hypoechoic uneven echo->benign\n\
    # taller-than-wide nodule, clear margin, regular shape, anechoic echo->benign\n\
    # wider-than-tall nodule, less clear margin, irregular shape, uneven echo->malignant\n\
    # Question: wider-than-tall nodule, unclear margin, irregular shape->",
    #     max_length=200,
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     pad_token_id=model.tokenizer.eos_token_id,
    # )
    # for seq in sequences:
    #     print(f"Result: {seq['generated_text']}")

    # model.to(device)

    # chkpt_dir = "/export/home/wuyueting/Encoder/mae_like/mae_main_wyt_revised/checkpoints/Prefix_LLM_MLP_base/feature_unshuffle/thyroid_prefix-004.pt"
    
    # checkpoint = torch.load(chkpt_dir, map_location='cpu')
    # model.load_state_dict(checkpoint, strict=False)
    # load data
    # data = TextImageTokenDataset(data_path = "data/large_lcond_finetune_falcon-7b-instruct_feature_unshuffle_train.pkl")

    # img, lang, lang_mask = data[1]

    # print(model)
    # out = model(img.unsqueeze(0).to(device), lang.unsqueeze(0).to(device), lang_mask.unsqueeze(0).to(device))
    # print(out)

    # tokenizer = AutoTokenizer.from_pretrained("/export/home/wuyueting/ReferenceModel/LLM/falcon/falcon-7b/")
    # model =  AutoModelForCausalLM.from_pretrained("/export/home/wuyueting/ReferenceModel/LLM/falcon/falcon-7b/")
    
