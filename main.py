import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def poincare_distance(u, v):
    """
    Calculate the Poincaré distance between two points in the hyperbolic space.
    
    Args:
        u (torch.Tensor): First point in the Poincaré disk
        v (torch.Tensor): Second point in the Poincaré disk
    
    Returns:
        torch.Tensor: The Poincaré distance between points u and v
    
    Raises:
        ValueError: If either point lies outside the unit disk
    """
    norm_u = u.norm(p=2)
    norm_v = v.norm(p=2)

    if norm_u >= 1 or norm_v >= 1:
        raise ValueError("Embeddings must be within the unit disk")
        
    diff = u - v
    distance = torch.acosh(1 + 2 * (diff.norm(p=2) ** 2) / ((1 - norm_u ** 2) * (1 - norm_v ** 2)))
    return distance

class PoincareEmbedding(nn.Module):
    """
    Neural network module for learning Poincaré embeddings.
    
    Args:
        num_embeddings (int): Number of embeddings to create
        embedding_dim (int): Dimension of each embedding vector
    
    Attributes:
        embeddings (nn.Parameter): Learnable embedding vectors
    """
    def __init__(self, num_embeddings, embedding_dim):
        super(PoincareEmbedding, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.5)

    def forward(self):
        """
        Forward pass of the model.
        
        Returns:
            torch.Tensor: The current embedding vectors
        """
        return self.embeddings

    def loss(self, positive_pairs):
        """
        Calculate the loss for the current embeddings based on positive pairs.
        
        Args:
            positive_pairs (list): List of tuples containing indices of points that 
                                 should be close to each other
        
        Returns:
            torch.Tensor: Average loss value for the given pairs
        """
        loss_value = 0.0
        for (i, j) in positive_pairs:
            loss_value += poincare_distance(self.embeddings[i], self.embeddings[j])
        return loss_value / len(positive_pairs)

def train(model, data_loader, num_epochs=150, lr=0.0001):
    """
    Train the Poincaré embedding model.
    
    This function handles the training loop, optimization, and visualization of the
    training progress. It also ensures that embeddings stay within the unit disk
    after each optimization step.
    
    Args:
        model (PoincareEmbedding): The model to train
        data_loader (list): List of positive pairs for training
        num_epochs (int, optional): Number of training epochs. Defaults to 150
        lr (float, optional): Learning rate for optimization. Defaults to 0.0001
    
    Returns:
        None: The function plots the training loss curve and updates the model in-place
    """
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
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
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
    """
    Visualize the learned Poincaré embeddings.
    
    Creates a scatter plot of the embeddings in 2D space, showing their positions
    within the unit disk (represented by a dashed red circle).
    
    Args:
        model (PoincareEmbedding): The model containing learned embeddings
    
    Returns:
        None: The function creates and displays a plot
    """
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
