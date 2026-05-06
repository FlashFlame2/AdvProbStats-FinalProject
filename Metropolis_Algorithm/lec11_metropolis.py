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


num_samples = 10000
initial_x = 0
stepsize = 1

samples = metropolis_sampler(num_samples, initial_x, stepsize)

x_range = np.linspace(-4, 4, 200)
Z = np.sqrt(2 * np.pi)
target_density = target_fn(x_range) / Z

plt.figure(figsize=(8, 5))
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Metropolis Samples')
plt.plot(x_range, target_density, 'orange', linewidth=2, label='Target Distribution = f(x)/Z')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Sampling with Metropolis Algorithm')
plt.legend()
plt.xlim(-4, 4)
plt.show()