import torch
import torch.optim as optim

def train(model, data_loader, num_epochs=200, lr=0.001):
    """Train the Poincaré embeddings model."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            
            loss = model.loss(batch)
            
            if torch.isnan(loss):
                print(f"NaN loss encountered at epoch {epoch+1}")
                return
            
            riemann_grad = model.riemannian_gradient(batch)
            
            with torch.no_grad():
                model.embeddings.data -= lr * riemann_grad
                
                norms = model.embeddings.norm(p=2, dim=1)
                mask = norms > 0.9  
                if mask.any():
                    model.embeddings.data[mask] = model.embeddings.data[mask] / norms[mask].view(-1, 1) * 0.9

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
