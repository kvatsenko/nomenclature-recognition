import numpy as np
import matplotlib.pyplot as plt

# --- Графік P(x) ---

# Задані параметри
Pa = 3500  # мм рт. ст.
Pb = 1500  # мм рт. ст.
L = 600    # мкм

# Генерація значень x
x = np.linspace(0, L, 100)

# Розрахунок P(x)
P = Pa - (Pa - Pb) * x / L

# Побудова графіка
plt.plot(x, P)
plt.title('Графік P(x)')
plt.xlabel('x, мкм')
plt.ylabel('P, мм рт. ст.')
plt.grid(True)
plt.show()

# --- Графік Q(x) ---

# Задані параметри
Pa = 3400      # мм рт. ст.
Pb = 12        # мм рт. ст.
L = 600e-6     # м
R = 3e-6       # м
η = 0.0012     # Па·с

# Генерація значень x
x = np.linspace(0, L, 100)

# Розрахунок градієнту тиску
dP_dx = (Pb - Pa) / L

# Розрахунок Q(x)
W = (8 * η) / (np.pi * R**4)
Q = -dP_dx / W * np.ones_like(x)

# Побудова графіка
plt.plot(x * 1e6, Q)
plt.title('Графік Q(x)')
plt.xlabel('x, мкм')
plt.ylabel('Q, м³/с')
plt.grid(True)
plt.show()

# --- Графік q(x) ---

# Задані параметри
Pa = 3400      # мм рт. ст.
Pb = 12        # мм рт. ст.
P0 = 20        # мм рт. ст.
L = 600e-6     # м
R = 3e-6       # м
η = 0.0012     # Па·с

# Генерація значень x
x = np.linspace(0, L, 100)

# Розрахунок P(x)
P = Pa - (Pa - Pb) * x / L

# Розрахунок q(x)
W = (8 * η) / (np.pi * R**4)
q = (P - P0) / W

# Побудова графіка
plt.plot(x * 1e6, q)
plt.title('Графік q(x)')
plt.xlabel('x, мкм')
plt.ylabel('q, м³/(с·м)')
plt.grid(True)
plt.show()

# --- Вплив Pa на K ---

# Задані параметри
Pb = 12       # мм рт. ст.
L = 600       # мкм
Norma = 20    # мм рт. ст.

# Значення Pa та K
Pa_values = [30.9, 33.5, 35.5]
K_values = [1.85, 2.78, 3.89]

# Генерація значень x
x = np.linspace(0, L, 100)

# Побудова графіків
plt.figure()
for Pa, K in zip(Pa_values, K_values):
    P = Pa - (Pa - Pb) * x / L
    plt.plot(x, P, label=f'Pa = {Pa}, K = {K}')

plt.axhline(Norma, color='red', linestyle='--', label='Норма')
plt.title('Вплив Pa на K')
plt.xlabel('x, мкм')
plt.ylabel('P, мм рт. ст.')
plt.legend()
plt.grid(True)
plt.show()

# --- Вплив P0 на K ---

# Задані параметри
Pb = 12       # мм рт. ст.
L = 600       # мкм
Norma = 20    # мм рт. ст.

# Значення P0 та K
P0_values = [22.5, 19.5, 16.5]
K_values = [0.328, 1.04, 7.923]

# Генерація значень x
x = np.linspace(0, L, 100)

# Побудова графіків
plt.figure()
for P0, K in zip(P0_values, K_values):
    P = P0 - (P0 - Pb) * x / L
    plt.plot(x, P, label=f'P0 = {P0}, K = {K}')

plt.axhline(Norma, color='red', linestyle='--', label='Норма')
plt.title('Вплив P0 на K')
plt.xlabel('x, мкм')
plt.ylabel('P, мм рт. ст.')
plt.legend()
plt.grid(True)
plt.show()

# --- Вплив r на P(x) ---

# Функція P(x)
def P_x(x, P0, Pb, Pa, L, λ):
    A = ((P0 - Pb) + (Pa - P0) * np.exp(-L / λ)) / (np.exp(-L / λ) - np.exp(L / λ))
    B = ((Pb - P0) + (P0 - Pa) * np.exp(L / λ)) / (np.exp(-L / λ) - np.exp(L / λ))
    return A * np.exp(x / λ) + B * np.exp(-x / λ) + P0

# Задані параметри
P0 = 25.0     # мм рт. ст.
Pb = 12.0     # мм рт. ст.
Pa = 60.0     # мм рт. ст.
L = 600.0     # мкм
l = 0.6       # мкм
R = 3.0       # мкм
N = 1.3       # м^-2
η = 0.09      # Па·с

# Значення r
r_values = [0.001, 0.05, 0.1, 0.2]  # мкм

# Генерація значень x
x = np.linspace(0, L, 100)

# Побудова графіків
plt.figure()
for r in r_values:
    λ = np.sqrt((R**3 * l) / (2 * np.pi * (r**4) * N))
    P = P_x(x, P0, Pb, Pa, L, λ)
    plt.plot(x, P, label=f'r = {r} мкм')

plt.title('Вплив r на P(x)')
plt.xlabel('x, мкм')
plt.ylabel('P(x), мм рт. ст.')
plt.legend()
plt.grid(True)
plt.show()
