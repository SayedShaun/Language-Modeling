from typing import Any, List, Union
from matplotlib import pyplot as plt
import numpy as np
from torch import nn, Tensor
from tqdm import tqdm
import torch
from dataloader import Tokenizer, get_dataloader
from modules import Block, Config, LayerNorm, Linear, PositionalEmbeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyGPT(nn.Module):
    def __init__(self, config:Config):
        super(TinyGPT, self).__init__()
        self.embedding = PositionalEmbeddings(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = LayerNorm(config.d_model)
        self.linear = Linear(config.d_model, config.vocab_size)
        self.loss = []

    def forward(self, X:Tensor, mask:Tensor=None, return_last_state:bool=False)->Tensor:
        X = self.embedding(X)
        for block in self.blocks:
            X = block(X, mask)
        X = self.norm(X)
        logits = self.linear(X)
        if return_last_state:
            return X
        return logits
    
    def _make_causal_mask(self, X:Tensor)->Tensor:
        mask = torch.tril(torch.ones(X.shape[1], X.shape[1]))
        return mask

    def trainable_parameters(self):
        trainable_params = 0
        for p in self.parameters():
            if p.requires_grad:
                trainable_params += p.numel()
        return f"Total Trainable Params: {trainable_params/1e6:.2f} M"
    
    @torch.no_grad()
    def generate(self, start:str, max_length:int=50, temperature:float=0.5)->str:
        start = Tokenizer.encode(start)
        outputs = start
        for _ in range(max_length):
            long = torch.LongTensor(outputs).unsqueeze(0).to(device)
            logits = self(long)[:, -1, :]/temperature
            probs = torch.softmax(logits, dim=-1)
            index = torch.multinomial(probs, num_samples=1)
            top_p = index[0, -1].item()
            outputs.append(top_p)
        return "".join(Tokenizer.decode(outputs))
    
    def train(self, epochs, dataloader, optimizer, lr=1e-3, verbose=False):
        optim_dict = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
            "adadelta": torch.optim.Adadelta,
            "adamw": torch.optim.AdamW,
            "adamax": torch.optim.Adamax,
            "asgd": torch.optim.ASGD
            }
        if optimizer not in optim_dict:
            raise ValueError(f"Optimizer {optimizer} is not supported. Please choose from {optim_dict.keys()}")
        optimizer = optim_dict[optimizer]
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optimizer(model.parameters(), lr=lr)
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.0
            for batch, (input_ids, target) in enumerate(dataloader):
                input_ids = input_ids.to(device)
                target = target.to(device)
                mask = self._make_causal_mask(input_ids).to(device)
                
                optimizer.zero_grad()
                logits = self(input_ids, mask)
                logits = logits.reshape(-1, logits.shape[-1])
                loss = loss_fn(logits, target.reshape(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss/batch
            self.loss.append(avg_loss)

            if verbose:
                print("Loss: ", avg_loss)
    
    def plot_history(self):
        plt.plot(self.loss)
        plt.show()

    def get_embeddings(self, text: Union[str, List[str]], return_type:str="np")->Union[torch.Tensor, List, np.ndarray]:
        input_ids = Tokenizer.encode(text)
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        hidden_states = model(input_ids, return_hidden_states=True)
        out_embed = hidden_states.mean(1).detach().cpu().numpy()

        valid_types = ["torch", "np", "list"]
        if return_type not in valid_types:
            raise ValueError(f"return_type must be one of {valid_types}")
        
        match return_type:
            case "torch":
                return torch.tensor(out_embed)
            case "list":
                return out_embed.tolist()[0]
        return out_embed
    
if __name__ == "__main__":
    # Loading trained model checkpoint
    config = Config
    model = TinyGPT(config)

    # dataloader = get_dataloader("Data/Sherlock Holmes.txt")
    # config = Config(
    #     vocab_size=Tokenizer.vocab_size(),
    #     n_layers=4,
    #     n_heads=4,
    #     d_model=256,
    #     d_ff=1024,
    #     max_len=1024,
    #     dropout=0.1
    # )
    # model = TinyGPT(config).to(device)
    # model.train(epochs=10, dataloader=dataloader, optimizer="adam", lr=1e-3, verbose=True)
    # torch.save(model.state_dict(), "Data/TinyGPT.tar")

    model.load_state_dict(torch.load("Data/TinyGPT.tar", map_location=device))
    response = model.generate("Sherlock Holmes", max_length=100, temperature=0.5)
    """
    Sherlock Holmes, I had been out. Then very carelessly scrap my own with a sens, and the chamber which consisted of the 
    home-crep to me over a strong and he stood, even the man of the drug a strong with a journey to see it is of the room." 
    "The name is the paper was the matter of the mask to addressing his armchair, and a seat to see, and looked his agent 
    a length of the deep harsh voice and a man of a capitald
    """

    # Lets check the embeddings
    embeddings = model.get_embeddings(["Sherlock Holmes", "Doctor Watson"])
    print(embeddings.shape) #-> (2, 256)        