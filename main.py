import torch
from torch.utils.data import DataLoader
from data import HyperlexDataset
from model import PoincareEmbedding
from train import train
from visualization import plot_embeddings

def main(file_path):
    dataset = HyperlexDataset(file_path)
    data_loader = DataLoader(dataset, 
                             batch_size=32, 
                             shuffle=True, 
                             collate_fn=dataset.collate_fn)

    model = PoincareEmbedding(num_embeddings=len(dataset.word_to_idx), embedding_dim=2)
    train(model, data_loader)
    plot_embeddings(model, dataset)

if __name__ == "__main__":
    file_path = 'hyperlex-all.txt' 
    main(file_path)
