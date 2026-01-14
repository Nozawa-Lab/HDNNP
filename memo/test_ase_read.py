from ase.io import read

atoms = read("OUTCAR", index=-1, format="vasp-out")

print(len(atoms))
print(atoms.get_chemical_symbols())
print(atoms.get_atomic_numbers())
print(atoms.get_scaled_positions())
print(atoms.get_volume())