import sympy as sp

# Paramètre symbolique
p, k = sp.symbols('p k', integer=True)

# Matrice 4x4 dépendant d'un seul paramètre
M = sp.Matrix([
    [1-p,   p/2,   p/2,     0],
    [p/2,   1-p,   0,     p/2],
    [p/2,   0,     1-p,   p/2],
    [0,     p/2,   p/2,   1-p]
])

# Diagonalisation
P, D = M.diagonalize()

# Inverse de la matrice de passage
P_inv = P.inv()

print("Matrice de passage P :")
sp.pprint(P)

print("\nMatrice diagonale D :")
sp.pprint(D)

print("\nP^{-1} =")
sp.pprint(P_inv)

# D^k
Dk = D.applyfunc(lambda x: x**k)
print("\nD^k =")
sp.pprint(Dk)

# Vérification : P^{-1} M P = D
print("\nCheck P^{-1} M P = D ?")
sp.pprint(sp.simplify(P * Dk * P_inv))

v = sp.Matrix([1, 0, 0, 0])

# Compute (P D^k P^{-1}) v safely
result = P * Dk * P_inv * v

result = sp.simplify(result)
sp.pprint(result)