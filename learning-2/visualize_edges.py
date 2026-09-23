import numpy as np
import matplotlib.pyplot as plt

def plot_edge_curves(layer, layer_idx=0):
    """
    Plots the learned polynomial curve for every input-to-output edge in a KumoLayer.
    """
    in_dim = layer.in_features
    out_dim = layer.out_features
    degree = layer.degree
    
    x_range = np.linspace(-2.0, 2.0, 200)
    
    basis = np.ones((len(x_range), degree + 1))
    for d in range(1, degree + 1):
        basis[:, d] = x_range ** d
        
    fig, axes = plt.subplots(in_dim, out_dim, figsize=(3 * out_dim, 3 * in_dim), squeeze=False)
    fig.suptitle(f"KumoLayer {layer_idx + 1} - Learned Edge Curves φ(x)", fontsize=14)
    
    for i in range(in_dim):
        for o in range(out_dim):

            coeffs = layer.C[i, o, :]
            y_curve = np.dot(basis, coeffs)
            
            ax = axes[i, o]
            ax.plot(x_range, y_curve, color='crimson', linewidth=2)
            ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax.axvline(0, color='black', linestyle='--', alpha=0.3)
            ax.set_title(f"Edge: Input {i} → Node {o}")
            ax.grid(True, linestyle=':', alpha=0.6)
            
    plt.tight_layout()
    plt.savefig(f"kumo_layer_{layer_idx + 1}_edges.png", dpi=300)
    print(f"Saved curve plot as 'kumo_layer_{layer_idx + 1}_edges.png'")
    plt.show()