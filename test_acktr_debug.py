"""Debug script to check ACKTR implementation"""
import torch
from stable_baselines3.acktr.kfac import KFACOptimizer
import torch.nn as nn

# Test basic K-FAC optimizer setup
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
    
    def forward(self, x):
        return self.linear(x)

net = SimpleNet()
optimizer = KFACOptimizer(
    net,
    lr=0.25,
    momentum=0.9,
    stat_decay=0.99,
    damping=1e-2,
)

print("K-FAC Optimizer initialized successfully")
print(f"Known modules: {len(optimizer.known_modules)}")
print(f"Modules: {optimizer.modules}")
print(f"Update frequency: {optimizer.update_freq}")
print(f"Damping: {optimizer.damping}")

# Test forward pass
x = torch.randn(10, 4)
y = net(x)
loss = y.sum()
loss.backward()

print("\nGradients computed")
print(f"Linear weight grad shape: {net.linear.weight.grad.shape}")
print(f"Linear weight grad norm: {net.linear.weight.grad.norm():.4f}")

# Test optimizer step
optimizer.step()
print("\nOptimizer step completed")
