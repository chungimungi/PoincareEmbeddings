import torch
from torch.utils.data import Dataset

class HyperlexDataset(Dataset):
    def __init__(self, file_path, min_score=4.0, word_to_idx=None):
        self.pairs = []
        self.word_to_idx = word_to_idx if word_to_idx else {}
        self.load_data(file_path, min_score)

    def load_data(self, file_path, min_score):
        with open(file_path, 'r') as f:
            next(f)  # Skip the header line
            for line in f:
                columns = line.strip().split()
                word1, word2, pos, rel_type, avg_score = columns[:5]
                avg_score = float(avg_score)
                
                if rel_type.startswith('hyp') and avg_score >= min_score:
                    if word1 not in self.word_to_idx:
                        self.word_to_idx[word1] = len(self.word_to_idx)
                    if word2 not in self.word_to_idx:
                        self.word_to_idx[word2] = len(self.word_to_idx)
                    self.pairs.append((self.word_to_idx[word1], self.word_to_idx[word2], avg_score))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def collate_fn(self, batch):
        """Custom collate function to properly batch the pairs."""
        i = torch.tensor([item[0] for item in batch])
        j = torch.tensor([item[1] for item in batch])
        scores = torch.tensor([item[2] for item in batch])
        return i, j, scores
