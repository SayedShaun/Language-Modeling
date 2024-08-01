# TinyGPT

Welcome to TinyGPT, a fully-customizable Language Model built from scratch in PyTorch. This project aims to provide an easy-to-use and highly-customizable implementation of TinyGPT, allowing you to train your own  and explore the capabilities of the transformer architecture.

## Example

    config = Config
    model = TinyGPT(config)

    dataloader = get_dataloader("Data/Sherlock Holmes.txt")
    config = Config(
        vocab_size=Tokenizer.vocab_size(),
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        max_len=1024,
        dropout=0.1
    )
    model = TinyGPT(config).to(device)
    model.train(epochs=10, dataloader=dataloader, optimizer="adam", lr=1e-3, verbose=True)

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

## Key Features

- **Fully Customizable**: TinyGPT allows you to customize almost every aspect of the model, including the model architecture, hyperparameters, and training settings.
- **Easy to Use**: With a simple and intuitive API, you can quickly train your own GPT model and start exploring text generation and text to embeddings.
- **Fast and Efficient**: TinyGPT is designed to be fast and efficient, making it suitable for both small-scale and large-scale training tasks. It support both traditional attention and pytorch flash attention.

## Getting Started

To get started with TinyGPT, you can follow these steps:

1. Install the required dependencies:
2. Tune the hyperparameters as needed
3. Train the model

## Limitations
The 'Tokenizer' class only supports txt files. If you have other file types, please convert them to txt format.
   
