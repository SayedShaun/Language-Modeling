from typing import Any, Generator, List, Optional, Text
import sentencepiece as spm
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import glob


class Tokenizer:
   
    @staticmethod
    def train_vocab_model(input_data:Text, vocab_size:int=5000)->None:
        spm.SentencePieceTrainer.Train(
            input=input_data, 
            model_prefix="VOCAB_MODEL",
            vocab_size=vocab_size, 
            )
    @classmethod
    def load_model(cls, model_path="Data/VOCAB_MODEL.model")->None:
        if not glob.glob(model_path):
            raise ValueError("Model not found. Please train the model first.")
        sp = spm.SentencePieceProcessor(model_file=model_path)
        return sp

    @classmethod    
    def encode(cls, text: str, return_type:str="list")->List:
        sp = cls.load_model()
        tokens = sp.EncodeAsIds(text)
        return cls._convert(inputs=tokens, return_type=return_type)
        
    @classmethod
    def decode(cls, tokens:Optional[np.ndarray])->str:
        sp = cls.load_model()
        string = sp.DecodeIds(tokens)
        return string
    
    @classmethod
    def build_train_data(cls, file_path:Text, n_ctx:int = 512, 
        truncate:int=None, return_type:str="list")->Generator:

        with open(file_path, "r", encoding="utf-8") as file:
            texts = file.read()[:truncate]

        sp = cls.load_model()
        all_ids = cls._convert(sp.EncodeAsIds(texts), return_type)
        for i in range(0, len(all_ids) - n_ctx):
            input = all_ids[i: i + n_ctx]
            target = all_ids[i + 1: i + n_ctx + 1]
            yield input, target

    @classmethod
    def _convert(cls, inputs:List, return_type:str)->Any:
        valid_types = ["torch", "tf", "np", "list"]
        if return_type not in valid_types:
            raise ValueError(f"inputs must be one of {valid_types}")
        match return_type:
            case "torch":
                torch = eval("torch")
                return torch.tensor(inputs)
            case "tf":
                tf = eval("tf")
                return tf.convert_to_tensor(inputs)
            case "np":
                np = eval("np")
                return np.array(inputs)
        return inputs

    @staticmethod
    def vocab_size()->int:
        sp = Tokenizer.load_model()
        return sp.GetPieceSize()


def get_dataloader(
        data_path:Text, 
        return_type:str="torch", 
        batch_size:int=32, 
        shuffle:bool=False, 
        truncate:int=None
        )->DataLoader:
    data = Tokenizer.build_train_data(
        data_path, 
        return_type=return_type, 
        truncate=truncate
        )
    X, Y = zip(*data)
    dataset = TensorDataset(
        torch.stack(X), 
        torch.stack(Y)
        )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
        )
    return dataloader
    

if __name__ == "__main__":
    tokens = Tokenizer.encode("hello", return_type="list")
    print(tokens)


