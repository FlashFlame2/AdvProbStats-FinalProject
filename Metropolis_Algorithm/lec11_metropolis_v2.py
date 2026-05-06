import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def target_fn(x):
    return np.exp(-x**2 / 2)

def symmetric_proposal(current_x, stepsize=1):
    updated_x = current_x + np.random.normal(0, stepsize)
    return updated_x

def metropolis_sampler(num_samples, initial_x, stepsize=1):
    samples = [initial_x]
    x = initial_x
    accepted = 0
    for _ in range(num_samples - 1):
        x_proposal = symmetric_proposal(x, stepsize)
        acceptance_ratio = target_fn(x_proposal) / target_fn(x)
        if np.random.rand() < min(1, acceptance_ratio):
            x = x_proposal
            accepted += 1
        samples.append(x)
    
    acceptance_rate = accepted / (num_samples - 1)
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    return np.array(samples)

configs = [
    {'num_samples': 1000,  'initial_x': 0,  'stepsize': 1},
    {'num_samples': 1000,  'initial_x': 0,  'stepsize': 2},
    {'num_samples': 1000,  'initial_x': 0,  'stepsize': 5},
    {'num_samples': 10000, 'initial_x': 0,  'stepsize': 1},
    {'num_samples': 10000, 'initial_x': 5,  'stepsize': 1},
    {'num_samples': 10000, 'initial_x': -5, 'stepsize': 1},
    {'num_samples': 50000, 'initial_x': 0,  'stepsize': 0.5},
    {'num_samples': 50000, 'initial_x': 0,  'stepsize': 1},
    {'num_samples': 50000, 'initial_x': 0,  'stepsize': 3},
]

x_range = np.linspace(-4, 4, 200)
Z = np.sqrt(2 * np.pi)
target_density = target_fn(x_range) / Z

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for idx, cfg in enumerate(configs):
    samples = metropolis_sampler(cfg['num_samples'], cfg['initial_x'], cfg['stepsize'])
    ax = axes[idx]
    ax.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
    ax.plot(x_range, target_density, 'orange', linewidth=2, label='Target')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title(f"n={cfg['num_samples']}, x0={cfg['initial_x']}, step={cfg['stepsize']}")
    ax.set_xlim(-4, 4)
    ax.legend(fontsize=8)

plt.suptitle('Metropolis Algorithm - Parameter Comparison', fontsize=14)
plt.tight_layout()
plt.show()