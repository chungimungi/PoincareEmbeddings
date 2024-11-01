import torch
import torch.nn as nn

def poincare_distance(u, v):
    """Compute Poincaré distance between two points."""
    norm_u = u.norm(p=2)
    norm_v = v.norm(p=2)

    if norm_u >= 1 or norm_v >= 1:
        raise ValueError("Embeddings must be within the unit disk")
        
    diff = u - v
    distance = torch.acosh(1 + 2 * (diff.norm(p=2) ** 2) / ((1 - norm_u ** 2) * (1 - norm_v ** 2)))
    return distance

class PoincareEmbedding(nn.Module):
    """Neural network module for Poincaré embeddings."""
    def __init__(self, num_embeddings, embedding_dim):
        super(PoincareEmbedding, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.1)

    def forward(self):
        """Return embeddings."""
        return self.embeddings

    def loss(self, batch):
        """Calculate loss for given batch of pairs."""
        i, j, scores = batch
        loss_value = 0.0
        for idx in range(len(i)):
            loss_value += scores[idx] * poincare_distance(self.embeddings[i[idx]], self.embeddings[j[idx]])
        return loss_value / len(i)

    def riemannian_gradient(self, batch):
        """Calculate Riemannian gradient for optimization."""
        i, j, scores = batch
        grad = torch.zeros_like(self.embeddings)
        
        for idx in range(len(i)):
            u = self.embeddings[i[idx]]
            v = self.embeddings[j[idx]]
            try:
                distance = poincare_distance(u, v)
                if distance == 0:
                    continue
                
                term_u = scores[idx] * (u - v) / distance
                term_v = scores[idx] * (v - u) / distance
                
                grad[i[idx]] += term_u
                grad[j[idx]] += term_v
            except ValueError:
                continue
        
        return grad / len(i)
