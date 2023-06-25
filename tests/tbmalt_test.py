from tbmalt import Geometry, Periodic, Coulomb
import torch
cell = torch.tensor([[2., 0., 0.], [0., 4., 0.], [0., 0., 2.]])
pos = torch.tensor([[0., 0., 0.], [0., 2., 0.]])
num = torch.tensor([1, 1])
cutoff = torch.tensor([9.98])
system = Geometry(num, pos, cell, units='a')
periodic = Periodic(system, system.cells, cutoff=cutoff)
coulomb = Coulomb(system, periodic, method='search')
print(coulomb.invrmat)

device = "cuda:0"
"""Test Ewald summation for ch4 with 3d pbc."""
latvec = torch.tensor([[4., 4., 0.], [0., 4., 4.], [4., 0., 4.]],
                        device=device)
positions = torch.tensor([
[3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
[3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
cutoff = torch.tensor([9.98], device=device)
system = Geometry(numbers, positions, latvec, units='a')
periodic = Periodic(system, system.cells, cutoff=cutoff)
coulomb = Coulomb(system, periodic, method='search')
print(coulomb.invrmat)
