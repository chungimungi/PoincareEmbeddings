import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.5)

    def forward(self):
        """Return embeddings."""
        return self.embeddings

    def loss(self, positive_pairs):
        """Calculate loss for given positive pairs."""
        loss_value = 0.0
        for (i, j) in positive_pairs:
            loss_value += poincare_distance(self.embeddings[i], self.embeddings[j])
        return loss_value / len(positive_pairs)

    def riemannian_gradient(self, positive_pairs):
        """Calculate Riemannian gradient for optimization."""
        grad = torch.zeros_like(self.embeddings)
        
        for (i, j) in positive_pairs:
            u = self.embeddings[i]
            v = self.embeddings[j]
            distance = poincare_distance(u, v)
            if distance == 0:
                continue
            
            term_u = (u - v) / distance
            term_v = (v - u) / distance
            
            grad[i] += term_u
            grad[j] += term_v
        
        return grad / len(positive_pairs)

def train(model, data_loader, num_epochs=200, lr=0.001):
    """Train the Poincaré embeddings model."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for positive_pairs in data_loader:
            optimizer.zero_grad()
            loss = model.loss(positive_pairs)
            if torch.isnan(loss): 
                print(f"NaN loss encountered at epoch {epoch+1}")
                return
            
            riemann_grad = model.riemannian_gradient(positive_pairs)
            
            with torch.no_grad():
                model.embeddings -= lr * riemann_grad
            
                norms = model.embeddings.norm(p=2, dim=1)
                model.embeddings[norms > 1] /= norms[norms > 1].view(-1, 1)

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def plot_embeddings(model):
    embeddings = model.forward().detach().numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.title('Poincaré Embeddings Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='dashed')
    plt.gca().add_artist(circle)

    plt.grid()
    plt.axis('equal')
    plt.show()

positive_pairs = [(0, 1), (1, 2), (2, 3)]
data_loader = [positive_pairs]

model = PoincareEmbedding(num_embeddings=100, embedding_dim=2)
train(model, data_loader)
plot_embeddings(model)
