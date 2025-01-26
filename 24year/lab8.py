import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Параметри
P0 = 100  # мм рт. ст.
eta = 0.005  # Па·с
D = 1.0  # см
L = 10.0  # см (довжина судини)
l_over_L = 0.2
l = l_over_L * L
w_over_W = l_over_L
d_over_D_values = [0.0, 0.2, 0.4, 0.6]

# Розрахунки
w_over_W = w_over_W
delta_P_ab = -P0 / (2 + w_over_W)
P_ab_list = []
P_bc_list = []
P0_prime_list = []
delta_P_bc_list = []
P_bc_end_list = []

for d_over_D in d_over_D_values:
    D_d = D - d_over_D * D
    ratio = (D / D_d) ** 4
    w_prime_over_W = w_over_W * ratio
    P0_prime = P0 * (2 + w_prime_over_W) / (2 + w_over_W)
    delta_P_bc = -P0 * (w_prime_over_W) / (2 + w_over_W)
    P_ab = P0_prime + delta_P_ab
    P_bc = P_ab + delta_P_bc

    P0_prime_list.append(P0_prime)
    delta_P_bc_list.append(delta_P_bc)
    P_ab_list.append(P_ab)
    P_bc_list.append(P_bc)

    # Побудова графіка
    x = [0, (L - l) / L, (L - l) / L, 1, 1]
    P = [P0_prime, P_ab, P_ab + delta_P_bc, P_bc, P_bc + delta_P_ab]

    plt.plot(np.array(x) * L, P, label=f'd/D = {d_over_D}')

# Налаштування графіка
plt.title('Залежність тиску P(x) вздовж судини')
plt.xlabel('Відстань вздовж судини (см)')
plt.ylabel('Тиск (мм рт. ст.)')
plt.legend()
plt.grid(True)
plt.show()

# Виведення таблиці результатів
print("d/D\tw'/W\tP0'\t\tΔP_ab\t\tP_ab\t\tΔP_bc\t\tP_bc")
for i in range(len(d_over_D_values)):
    print(f"{d_over_D_values[i]:.1f}\t{w_over_W * (D / (D - d_over_D_values[i] * D)) ** 4:.3f}\t"
          f"{P0_prime_list[i]:.2f}\t{delta_P_ab:.2f}\t{P_ab_list[i]:.2f}\t"
          f"{delta_P_bc_list[i]:.2f}\t{P_bc_list[i]:.2f}")



# Параметри
P0 = 30  # мм рт. ст.
eta = 0.005  # Па·с
D = 0.01  # см
L = 1.0  # см
d_over_D_values = [0.0, 0.1, 0.3, 0.7]
l_over_L_values = [0.05, 0.1, 0.2, 0.4, 0.8]

# Функція для обчислення об'ємної швидкості кровотоку
def compute_Q0(P0, W_total):
    return P0 / W_total

# Побудова графіків P(x) для заданої системи параметрів
for d_over_D in d_over_D_values:
    plt.figure()
    for l_over_L in l_over_L_values:
        l = l_over_L * L
        w_over_W = l_over_L
        D_d = D - d_over_D * D
        ratio = (D / D_d) ** 4
        w_prime_over_W = w_over_W * ratio
        W_total = 2 + w_over_W + w_prime_over_W
        Q0 = P0 / W_total

        # Точки для графіка
        x = [0, (L - l) / L, (L - l) / L, 1]
        P = [P0, P0 - Q0 * (2), P0 - Q0 * (2 + w_over_W), P0 - Q0 * W_total]

        plt.plot(np.array(x) * L, P, label=f'l/L = {l_over_L}')

    plt.title(f'Залежність P(x) вздовж мікросудини при d/D = {d_over_D}')
    plt.xlabel('Відстань вздовж судини (см)')
    plt.ylabel('Тиск (мм рт. ст.)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Побудова графіків зміни об'ємної швидкості кровотоку
plt.figure()
for l_over_L in l_over_L_values:
    Q0_list = []
    d_over_D_list = []
    for d_over_D in np.linspace(0, 0.7, 50):
        l = l_over_L * L
        w_over_W = l_over_L
        D_d = D - d_over_D * D
        ratio = (D / D_d) ** 4
        w_prime_over_W = w_over_W * ratio
        W_total = 2 + w_over_W + w_prime_over_W
        Q0 = P0 / W_total

        Q0_list.append(Q0)
        d_over_D_list.append(d_over_D)

    plt.plot(d_over_D_list, Q0_list, label=f'l/L = {l_over_L}')

plt.title('Залежність Q0 від d/D')
plt.xlabel('d/D')
plt.ylabel('Q0 (умовні одиниці)')
plt.legend()
plt.grid(True)
plt.show()




# Параметри
P0 = 30  # мм рт. ст.
D = 0.01  # см
L = 1.0  # см
eta = 0.005  # Па·с

d_over_D_values = [0.0, 0.1, 0.3, 0.7]
l_over_L_values = [0.05, 0.1, 0.2, 0.4, 0.8]

# Порожній список для зберігання результатів
results = []

for l_over_L in l_over_L_values:
    w_over_W = l_over_L
    for d_over_D in d_over_D_values:
        D_d = D - d_over_D * D
        ratio = (D / D_d) ** 4
        w_prime_over_W = w_over_W * ratio
        W_total_over_W = 2 + w_over_W + w_prime_over_W
        Q0 = P0 / W_total_over_W
        delta_P_ab = -Q0
        P_ab = P0 + delta_P_ab
        delta_P_bc = -Q0 * w_prime_over_W
        P_bc = P_ab + delta_P_bc

        results.append({
            'l/L': l_over_L,
            'd/D': d_over_D,
            'w/W': w_over_W,
            "w'/W": w_prime_over_W,
            'W_total/W': W_total_over_W,
            'Q0 (ум. од.)': Q0,
            'ΔP_ab (мм рт. ст.)': delta_P_ab,
            'P_ab (мм рт. ст.)': P_ab,
            'ΔP_bc (мм рт. ст.)': delta_P_bc,
            'P_bc (мм рт. ст.)': P_bc
        })

# Створення таблиці
df = pd.DataFrame(results)

# Виведення таблиці
print(df)


