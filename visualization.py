# visualization.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_embeddings(model, dataset):
    embeddings = model.forward().detach().numpy()
    
    norms = np.linalg.norm(embeddings, axis=1)
    max_norm = np.max(norms)
    colors = cm.viridis(norms / norms.max())
    
    padding = 0.1
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_facecolor('white')
    
    circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--', 
                       linewidth=2, alpha=0.5, zorder=1)
    ax.add_artist(circle)
    
    circle_fill = plt.Circle((0, 0), 1, color='red', fill=True, 
                           alpha=0.05, zorder=0)
    ax.add_artist(circle_fill)
    
    for (i, j, score) in dataset.pairs:
        if i < len(embeddings) and j < len(embeddings):
            u, v = embeddings[i], embeddings[j]
            plt.plot([u[0], v[0]], [u[1], v[1]], 'gray', linestyle='-', 
                    linewidth=0.5, alpha=0.3, zorder=2)
    
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors, 
                         alpha=0.75, s=60, edgecolor='k', zorder=3)

    margin = 10
    plt.xlim(-1 - margin, 1 + margin)
    plt.ylim(-1 - margin, 1 + margin)
    
    info_text = (
        f"Maximum embedding norm: {max_norm:.3f}\n"
        f"Valid region: unit disk (norm < 1)\n"
        f"Dashed line: boundary of valid region"
    )
    plt.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.title('Poincaré Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axis('equal')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Distance from origin (norm)')
    
    plt.show()
