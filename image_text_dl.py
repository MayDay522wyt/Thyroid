# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
from random import randint
import random

# 设置随机种子为 42
random.seed(42)

import PIL
import argparse
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import transformers

import os
import pandas as pd
import numpy as np

class TextImageDataset(Dataset):
    def __init__(self,
                 folder: str,
                 sub_folder:str,
                 text_mode:str = "sentence",
                 transform=None,
                 shuffle=False,
                 custom_tokenizer=None,
                 max_length = 40
                 ):                                
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(os.path.join(folder,sub_folder))
        """
        在 Python 中，* 是一个解包运算符（unpacking operator）。
        在使用 glob 函数时，我们希望将生成器对象解包成一个列表，以便我们可以访问其中的文件路径。
        因此，我们使用 [*path.glob('**/*.txt')] 表达式，其中 * 将生成器对象解包成一个列表。

        glob 是一个用于匹配文件路径的函数。在给定的路径中，glob 函数可以使用通配符模式进行文件名匹配，并返回所有匹配的文件列表。
        path.glob('**/*.txt') 表示搜索给定路径下所有子文件夹中的 .txt 文件。** 表示匹配任意数量的子文件夹，*.txt 表示匹配以 .txt 结尾的文件。
        类似地，path.glob('**/*.png')、path.glob('**/*.jpg')、path.glob('**/*.jpeg') 和 path.glob('**/*.bmp') 分别匹配所有子文件夹中的 .png、.jpg、.jpeg 和 .bmp 文件。
        glob 函数返回一个生成器对象，通过使用 [*...] 语法将其转换为列表。这样，text_files 变量将包含所有匹配的 .txt 文件路径列表，image_files 变量将包含所有匹配的图片文件路径列表。
        """
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        image_files = {image_file.stem: image_file for image_file in image_files}

        text_csv = os.path.join(folder, sub_folder+"_"+text_mode+".csv")
        data_frame = pd.read_csv(text_csv)
        lines = data_frame.shape[0]
        texts = {}
        for idx in range(lines):
            # print(data_frame[idx, 0])
            texts[data_frame.iloc[idx, 0]] = data_frame.iloc[idx, 1]

        keys = (image_files.keys() & texts.keys())

        self.max_length = max_length
        self.keys = list(keys)
        self.text_files = {k: v for k, v in texts.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}

        if transform == None:
            self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.22, 0.22, 0.22], std=[0.08, 0.08, 0.08])  # 标准化图像
            ])
        else:
            self.image_transform=transform

        self.ct = custom_tokenizer
        if self.ct == None:
            self.custom_tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
        elif self.ct == "rawtext":
            self.custom_tokenizer = None
        else: #gpt2
            self.custom_tokenizer = custom_tokenizer
            self.custom_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.keys)
    
    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    # def random_sample(self):
    #     return self.__getitem__(randint(0, self.__len__() - 1))

    # def sequential_sample(self, ind):
    #     if ind >= self.__len__() - 1:
    #         return self.__getitem__(0)
    #     return self.__getitem__(ind + 1)

    # def skip_sample(self, ind):
    #     if self.shuffle:
    #         return self.random_sample()
    #     return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]
        image_name = key

        description = text_file

        if self.ct == None:
            tokens = self.custom_tokenizer(description, return_tensors='pt',max_length=self.max_length ,padding="max_length",truncation=True ) 
            lang, lang_mask = tokens["input_ids"][0], tokens["attention_mask"][0]
        elif self.ct == "rawtext":
            lang = description
            lang_mask = None
        else:
            tokens = self.custom_tokenizer.encode_plus(description, add_special_tokens=True, padding='max_length', max_length=self.max_length, truncation=True, return_attention_mask=True)
            lang, lang_mask = torch.tensor(tokens["input_ids"]), torch.tensor(tokens["attention_mask"])
        
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return image_tensor, lang, lang_mask, image_file

class TextImageTokenDataset(Dataset):
    def __init__(self,
                 prefix_length: int = 10,
                 data_path="data/large_lcond_finetune_train.pkl",
                 ):                                
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
        """
        super().__init__()
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
            self.img_embeddings = all_data["img_embeddings"]
            self.lang_tokens = all_data["lang_tokens"]
            self.lang_masks = all_data["lang_masks"]
        self.prefix_length = prefix_length

    def __len__(self):
        return len(self.img_embeddings)
    
    def __getitem__(self, item: int):
        img_token = self.img_embeddings[item]
        lang_token = self.lang_tokens[item]
        lang_mask = self.lang_masks[item]
        return img_token,lang_token, lang_mask

    
from transformers import GPT2Tokenizer
if __name__ =="__main__":
    data = TextImageTokenDataset(data_path="data/large_vit_coco_gpt2_caption_train.pkl")
    a,b,c = data[0]
    print(len(data.img_embeddings))
    # dataset_train = TextImageDataset("/export/home/wuyueting/thyroid_data/CLIPdata_folder/CLIPdata_image_cut_BM_output/","train","feature",custom_tokenizer=GPT2Tokenizer.from_pretrained('gpt2'), max_length = 30)
    # img, lang, lang_mask, image_path = dataset_train[1]
    # print(image_path)

    # sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=32,
    #     num_workers=1,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # for (samples, texts, text_masks) in data_loader_train:
    #     print(samples.shape, texts.shape, text_masks.shape)
    #     break
    # model = models_mae.__dict__["lcond_MAE"]()
    # loss, _,_ = model(samples, texts, text_masks)
    # print(loss)

    # dataset = TextImageTokenDataset()
    # print(dataset[1])