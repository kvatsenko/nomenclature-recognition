# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint

# # Визначення диференціального рівняння
# def dxdt(x0, t, epsilon, sigma, gamma):
#     # numerator = x0 * epsilon
#     # denominator = (epsilon - sigma * x0) * np.exp(-epsilon * t) + sigma * x0
#     y = (x0 * epsilon) / ((epsilon - sigma * x0) * np.exp(-epsilon * t) + sigma * x0)
#     return y


# # print(2.718281**2)
# # Параметри
# epsilon = 0.5  # можна змінювати
# sigma_values = [0.002, 0.005, 0.01]  # різні значення σ
# gamma = 1  # можна змінювати
# x0_values = [5, 10, 15]  # різні значення x0

# # Часові значення
# t = np.linspace(0, 20)

# # Графік для різних σ
# plt.figure(figsize=(10, 6))

# for sigma in sigma_values:
#     x = odeint(dxdt, x0_values[0], t, args=(epsilon, sigma, gamma))
#     plt.plot(t, x, label=f'σ = {sigma}')

# # plt.title('Розв\'язок рівняння для різних σ')
# # plt.xlabel('Час (t)')
# # plt.ylabel('Чисельність популяції x(t)')
# # plt.legend()
# # plt.grid()
# # plt.show()
# plt.plot(t, y, label=r"$y=\frac{10 \cdot 0.5}{(0.5 - 0.002 \cdot 10) \cdot e^{-0.5 \cdot x} + 0.002 \cdot 10}$")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Plot of y as a function of x")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Графік для різних γ
# gamma_values = [0.5, 0.7, 1.5]
# plt.figure(figsize=(10, 6))

# for gamma in gamma_values:
#     x = odeint(dxdt, x0_values[0], t, args=(epsilon, sigma_values[0], gamma))
#     plt.plot(t, x, label=f'γ = {gamma}')

# plt.title('Розв\'язок рівняння для різних γ')
# plt.xlabel('Час (t)')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid()
# plt.show()

# # Графік для різних x0
# plt.figure(figsize=(10, 6))

# for x0 in x0_values:
#     x = odeint(dxdt, x0, t, args=(epsilon, sigma_values[0], gamma_values[0]))
#     plt.plot(t, x, label=f'x₀ = {x0}')

# plt.title('Розв\'язок рівняння для різних x₀')
# plt.xlabel('Час (t)')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid()
# plt.show()









# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint

# # Визначення диференціального рівняння
# def dxdt(x0, t, epsilon, sigma, gamma):
#     return np.exp(epsilon * t) * x0



# # Параметри
# epsilon = 0.5  # можна змінювати
# sigma_values = [0.1, 0.2, 0.3]  # різні значення σ
# gamma = 1  # можна змінювати
# x0_values = [5, 10, 15]  # різні значення x0

# # Часові значення
# t = np.linspace(0, 10, 100)

# # Графік для різних σ
# plt.figure(figsize=(10, 6))
# for sigma in sigma_values:
#     x = odeint(dxdt, x0_values[0], t, args=(epsilon, sigma, gamma))
#     plt.plot(t, x, label=f'σ = {sigma}')
# plt.title('Розв\'язок рівняння для різних σ')
# plt.xlabel('Час (t)')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid()
# plt.show()

# # Графік для різних γ
# gamma_values = [1, 2, 3]
# plt.figure(figsize=(10, 6))
# for gamma in gamma_values:
#     x = odeint(dxdt, x0_values[0], t, args=(epsilon, sigma_values[0], gamma))
#     plt.plot(t, x, label=f'γ = {gamma}')
# plt.title('Розв\'язок рівняння для різних γ')
# plt.xlabel('Час (t)')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid()
# plt.show()

# # Графік для різних x0
# plt.figure(figsize=(10, 6))
# for x0 in x0_values:
#     x = odeint(dxdt, x0, t, args=(epsilon, sigma_values[0], gamma_values[0]))
#     plt.plot(t, x, label=f'x₀ = {x0}')
# plt.title('Розв\'язок рівняння для різних x₀')
# plt.xlabel('Час (t)')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid()
# plt.show()

















# 1!
# import numpy as np
# import matplotlib.pyplot as plt

# # Функція для обчислення x(t)
# def x_t(t, x0, epsilon, sigma):
#     numerator = x0 * epsilon
#     denominator = (epsilon - sigma * x0) * np.exp(-epsilon * t) + sigma * x0
#     return numerator / denominator

# # Часовий інтервал
# t = np.linspace(0, 20, 1000)

# # Параметри
# x0_values = [5, 10, 15]  # Різні значення x₀
# epsilon_values = [0.5, 1.0, 1.5]  # Різні значення ε
# sigma_values = [0.002, 0.005, 0.01]  # Різні значення σ

# # 1. Графіки для різних значень ε при фіксованих σ та x₀
# sigma = 0.002  # Фіксуємо σ
# x0 = 10        # Фіксуємо x₀

# plt.figure(figsize=(10,6))
# for epsilon in epsilon_values:
#     y = x_t(t, x0, epsilon, sigma)
#     plt.plot(t, y, label=f'ε = {epsilon}')
# plt.xlabel("Час t")
# plt.ylabel("Чисельність популяції x(t)")
# plt.title("Залежність x(t) від часу при різних значеннях ε")
# plt.legend()
# plt.grid(True)
# plt.show()

# # 2. Графіки для різних значень σ при фіксованих ε та x₀
# epsilon = 0.5  # Фіксуємо ε
# x0 = 10        # Фіксуємо x₀

# plt.figure(figsize=(10,6))
# for sigma in sigma_values:
#     y = x_t(t, x0, epsilon, sigma)
#     plt.plot(t, y, label=f'σ = {sigma}')
# plt.xlabel("Час t")
# plt.ylabel("Чисельність популяції x(t)")
# plt.title("Залежність x(t) від часу при різних значеннях σ")
# plt.legend()
# plt.grid(True)
# plt.show()

# # 3. Графіки для різних значень x₀ при фіксованих ε та σ
# epsilon = 0.5  # Фіксуємо ε
# sigma = 0.002  # Фіксуємо σ

# plt.figure(figsize=(10,6))
# for x0 in x0_values:
#     y = x_t(t, x0, epsilon, sigma)
#     plt.plot(t, y, label=f'x₀ = {x0}')
# plt.xlabel("Час t")
# plt.ylabel("Чисельність популяції x(t)")
# plt.title("Залежність x(t) від часу при різних значеннях x₀")
# plt.legend()
# plt.grid(True)
# plt.show()




# 2!
# import numpy as np
# import matplotlib.pyplot as plt

# def x_t(t, x0, epsilon, sigma):
#     numerator = x0 * epsilon
#     denominator = (epsilon - sigma * x0) * np.exp(-epsilon * t) + sigma * x0
#     return numerator / denominator

# # Часовий інтервал
# t = np.linspace(0, 50, 1000)

# # Параметри
# x0 = 10
# epsilon = 0.4
# sigma_values = [0.002, 0.005, 0.01]

# # Побудова графіка
# plt.figure(figsize=(10, 6))
# for sigma in sigma_values:
#     y = x_t(t, x0, epsilon, sigma)
#     x_st = epsilon / sigma
#     plt.plot(t, y, label=f'σ = {sigma}, xₛₜ = {x_st}')
#     print(f'Для σ = {sigma}: xₛₜ = {x_st}')

# plt.title('Залежність x(t) при різних значеннях σ (x₀ та ε стабільні)')
# plt.xlabel('Час t')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid(True)
# plt.show()


# 3!
# x0 = 10
# sigma = 0.001
# epsilon_values = [1.0, 1.5, 2.0]

# # Побудова графіка
# plt.figure(figsize=(10, 6))
# for epsilon in epsilon_values:
#     y = x_t(t, x0, epsilon, sigma)
#     x_st = epsilon / sigma
#     plt.plot(t, y, label=f'ε = {epsilon}, xₛₜ = {x_st}')
#     print(f'Для ε = {epsilon}: xₛₜ = {x_st}')

# plt.title('Залежність x(t) при різних значеннях ε (x₀ та σ стабільні)')
# plt.xlabel('Час t')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid(True)
# plt.show()


# 4!
# epsilon = 0.6
# sigma = 0.01
# x0_values = [1, 5, 20, 40]

# # Побудова графіка
# plt.figure(figsize=(10, 6))
# for x0 in x0_values:
#     y = x_t(t, x0, epsilon, sigma)
#     x_st = epsilon / sigma
#     plt.plot(t, y, label=f'x₀ = {x0}, xₛₜ = {x_st}')
#     print(f'Для x₀ = {x0}: xₛₜ = {x_st}')

# plt.title('Залежність x(t) при різних значеннях x₀ (ε та σ стабільні)')
# plt.xlabel('Час t')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid(True)
# plt.show()




# # 5!
# import numpy as np
# import matplotlib.pyplot as plt

# # Параметри
# sigma = 0.01  # Фіксоване значення σ
# epsilon_values = np.linspace(0.1, 2.0, 100)  # Змінюємо ε від 0.1 до 2.0

# # Розрахунок x_st
# x_st_values = epsilon_values / sigma

# # Побудова графіка
# plt.figure(figsize=(10, 6))
# plt.plot(epsilon_values, x_st_values)
# plt.title('Залежність стаціонарного значення xₛₜ від ε при σ = 0.01')
# plt.xlabel('Коефіцієнт росту ε')
# plt.ylabel('Стаціонарне значення xₛₜ')
# plt.grid(True)
# plt.show()

# # 5.1!
# # Параметри
# epsilon = 1.0  # Фіксоване значення ε
# sigma_values = np.linspace(0.001, 0.02, 100)  # Змінюємо σ від 0.001 до 0.02

# # Розрахунок x_st
# x_st_values = epsilon / sigma_values

# # Побудова графіка
# plt.figure(figsize=(10, 6))
# plt.plot(sigma_values, x_st_values)
# plt.title('Залежність стаціонарного значення xₛₜ від σ при ε = 1.0')
# plt.xlabel('Коефіцієнт саморегуляції σ')
# plt.ylabel('Стаціонарне значення xₛₜ')
# plt.grid(True)
# plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def x_t(t, x0, epsilon, sigma):
    numerator = x0 * epsilon
    denominator = (epsilon - sigma * x0) * np.exp(-epsilon * t) + sigma * x0
    return numerator / denominator

# Функція для знаходження T_0.9
def find_T_09(x0, epsilon, sigma):
    x_st = epsilon / sigma
    x_target = 0.9 * x_st

    def equation(t):
        return x_t(t, x0, epsilon, sigma) - x_target

    T_09 = fsolve(equation, 1)[0]
    return T_09


# Т0,9 (х0)
# Параметри для дослідження залежності T_0.9 від x0
x0_values = np.linspace(1, 50, 50)
epsilon = 1.0
sigma = 0.01

T_09_values = []
for x0 in x0_values:
    T_09 = find_T_09(x0, epsilon, sigma)
    T_09_values.append(T_09)

# Побудова графіка T_0.9(x0)
plt.figure(figsize=(10, 6))
plt.plot(x0_values, T_09_values)
plt.title('Залежність T₀.₉ від x₀ при ε = 1.0, σ = 0.01')
plt.xlabel('Початкова чисельність x₀')
plt.ylabel('Час T₀.₉')
plt.grid(True)
plt.show()


# Т0,9 (ε)
# Параметри
x0 = 10     # Фіксоване x₀
sigma = 0.01  # Фіксоване σ
epsilon_values = np.linspace(0.1, 2.0, 100)  # Змінюємо ε від 0.1 до 2.0

T_09_values = []
for epsilon in epsilon_values:
    T_09 = find_T_09(x0, epsilon, sigma)
    T_09_values.append(T_09)

# Побудова графіка T₀.₉(ε)
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, T_09_values)
plt.title('Залежність T₀.₉ від ε при x₀ = 10, σ = 0.01')
plt.xlabel('Коефіцієнт росту ε')
plt.ylabel('Час T₀.₉')
plt.grid(True)
plt.show()

# Т0,9 (𝜎) 
# Параметри
x0 = 10     # Фіксоване x₀
epsilon = 1.0  # Фіксоване ε
sigma_values = np.linspace(0.001, 0.05, 100)  # Змінюємо σ від 0.001 до 0.05

T_09_values = []
for sigma in sigma_values:
    T_09 = find_T_09(x0, epsilon, sigma)
    T_09_values.append(T_09)

# Побудова графіка T₀.₉(σ)
plt.figure(figsize=(10, 6))
plt.plot(sigma_values, T_09_values)
plt.title('Залежність T₀.₉ від σ при x₀ = 10, ε = 1.0')
plt.xlabel('Коефіцієнт саморегуляції σ')
plt.ylabel('Час T₀.₉')
plt.grid(True)
plt.show()




# xₖ(ε)
def find_tk(x0, epsilon, sigma):
    x_k = epsilon / (2 * sigma)
    
    def equation(t):
        return x_t(t, x0, epsilon, sigma) - x_k

    t_k = fsolve(equation, 1)[0]
    return t_k, x_k

# Параметри
x_k_values = []
for epsilon in epsilon_values:
    _, x_k = find_tk(x0, epsilon, sigma)
    x_k_values.append(x_k)

# Побудова графіка xₖ(ε)
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, x_k_values)
plt.title('Залежність xₖ від ε при x₀ = 10, σ = 0.01')
plt.xlabel('Коефіцієнт росту ε')
plt.ylabel('Чисельність популяції xₖ')
plt.grid(True)
plt.show()


# xₖ(σ)
# Параметри
x_k_values = []
for sigma in sigma_values:
    _, x_k = find_tk(x0, epsilon, sigma)
    x_k_values.append(x_k)

# Побудова графіка xₖ(σ)
plt.figure(figsize=(10, 6))
plt.plot(sigma_values, x_k_values)
plt.title('Залежність xₖ від σ при x₀ = 10, ε = 1.0')
plt.xlabel('Коефіцієнт саморегуляції σ')
plt.ylabel('Чисельність популяції xₖ')
plt.grid(True)
plt.show()

# xₖ(x₀)
# Параметри
x_k_values = []
for x0 in x0_values:
    _, x_k = find_tk(x0, epsilon, sigma)
    x_k_values.append(x_k)

# Побудова графіка xₖ(x₀)
plt.figure(figsize=(10, 6))
plt.plot(x0_values, x_k_values)
plt.title('Залежність xₖ від x₀ при ε = 1.0, σ = 0.01')
plt.xlabel('Початкова чисельність x₀')
plt.ylabel('Чисельність популяції xₖ')
plt.grid(True)
plt.show()

# tₖ(ε)
# Вже маємо функцію find_tk
t_k_values = []
for epsilon in epsilon_values:
    t_k, _ = find_tk(x0, epsilon, sigma)
    t_k_values.append(t_k)

# Побудова графіка tₖ(ε)
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, t_k_values)
plt.title('Залежність tₖ від ε при x₀ = 10, σ = 0.01')
plt.xlabel('Коефіцієнт росту ε')
plt.ylabel('Час tₖ')
plt.grid(True)
plt.show()

# tₖ(σ)
t_k_values = []
for sigma in sigma_values:
    t_k, _ = find_tk(x0, epsilon, sigma)
    t_k_values.append(t_k)

# Побудова графіка tₖ(σ)
plt.figure(figsize=(10, 6))
plt.plot(sigma_values, t_k_values)
plt.title('Залежність tₖ від σ при x₀ = 10, ε = 1.0')
plt.xlabel('Коефіцієнт саморегуляції σ')
plt.ylabel('Час tₖ')
plt.grid(True)
plt.show()

# tₖ(x₀)
t_k_values = []
for x0 in x0_values:
    t_k, _ = find_tk(x0, epsilon, sigma)
    t_k_values.append(t_k)

# Побудова графіка tₖ(x₀)
plt.figure(figsize=(10, 6))
plt.plot(x0_values, t_k_values)
plt.title('Залежність tₖ від x₀ при ε = 1.0, σ = 0.01')
plt.xlabel('Початкова чисельність x₀')
plt.ylabel('Час tₖ')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

def population(epsilon, delta, x0, t_end, points=1000):
    t = np.linspace(0, t_end, points)
    number = (x0 * epsilon) / ((epsilon - delta * x0) * np.exp(-epsilon * t) + delta * x0)
    return t, number

def find_t(target_population, epsilon, delta, x0):
    func = lambda t: (x0 * epsilon) / ((epsilon - delta * x0) * np.exp(-epsilon * t) + delta * x0) - target_population
    return newton(func, x0=3, tol=1e-6)

# Параметри
epsilon = 0.4
delta = 0.002
x0 = 10

# Побудова графіка
t, res = population(epsilon, delta, x0, t_end=200)
plt.plot(t, res)
plt.xlabel("Time (t)")
plt.ylabel("Population")
plt.title("Population Growth Over Time")
plt.grid(True)
plt.show()

# Знаходження часу для заданого значення популяції
x_k = epsilon / (2 * delta)
t_k = find_t(x_k, epsilon, delta, x0)
print("Time t_k:", t_k)

# Зміна значень epsilon
epsilons = np.arange(0.4, 2, 0.6)
plt.show()















# import numpy as np
# import matplotlib.pyplot as plt

# # Параметри
# x0 = 10
# epsilon = 0.5
# sigma = 0.002

# # Діапазон значень x
# t = np.linspace(0, 20)

# # Обчислення y
# y = (x0 * epsilon) / ((epsilon - sigma * x0) * np.exp(-epsilon * t) + sigma * x0)

# # Побудова графіка
# plt.plot(t, y, label=r"$y=\frac{10 \cdot 0.5}{(0.5 - 0.002 \cdot 10) \cdot e^{-0.5 \cdot x} + 0.002 \cdot 10}$")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Plot of y as a function of x")
# plt.legend()
# plt.grid(True)
# plt.show()



















# # Визначення диференціального рівняння для моделі Мальтуса
# def dxdt(x, t, epsilon):
#     return epsilon * x

# # Параметри
# epsilon_values = [0.1, 0.2, 0.5]  # різні значення ε
# x0 = 10  # початкова чисельність популяції

# # Часові значення
# t_max = 50
# t = np.linspace(0, t_max, 1000)

# # Розрахунок та побудова графіків x(t) для різних ε
# plt.figure(figsize=(10, 6))
# T_half_analytical = []
# T_half_numerical = []

# for epsilon in epsilon_values:
#     x = odeint(dxdt, x0, t, args=(epsilon,))
#     x = x.flatten()
#     plt.plot(t, x, label=f'ε = {epsilon}')
    
#     # Аналітичний розрахунок T0.5
#     T_half = np.log(2) / epsilon
#     T_half_analytical.append(T_half)
    
#     # Чисельне визначення T0.5
#     x_target = 2 * x0
#     idx = np.where(x >= x_target)[0][0]
#     T_half_num = t[idx]
#     T_half_numerical.append(T_half_num)
    
#     print(f"Для ε = {epsilon}:")
#     print(f"Теоретичне T0.5 = {T_half:.4f}")
#     print(f"Чисельне T0.5 = {T_half_num:.4f}\n")


# # # Розрахунок похідної в момент часу t=0
# # for epsilon in epsilon_values:
# #     # Теоретична швидкість
# #     dxdt_analytical = epsilon * x0
    
# #     # Чисельна швидкість (наближене обчислення похідної)
# #     x = odeint(dxdt, x0, t, args=(epsilon,))
# #     x = x.flatten()
# #     dt = t[1] - t[0]
# #     dxdt_numerical = (x[1] - x[0]) / dt
    
# #     print(f"Для ε = {epsilon}:")
# #     print(f"Аналітична швидкість dx/dt|t=0 = {dxdt_analytical:.4f}")
# #     print(f"Чисельна швидкість dx/dt|t=0 ≈ {dxdt_numerical:.4f}\n")




# plt.title('Залежність чисельності популяції x(t) від часу для різних ε')
# plt.xlabel('Час (t)')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid()
# plt.show()

# # Побудова графіка залежності T0.5(ε)
# epsilon_range = np.linspace(0.05, 1, 100)
# T_half_values = np.log(2) / epsilon_range

# plt.figure(figsize=(10, 6))
# plt.plot(epsilon_range, T_half_values)
# plt.title('Залежність характерного часу T0.5 від коефіцієнта росту ε')
# plt.xlabel('Коефіцієнт росту ε')
# plt.ylabel('Характерний час T0.5')
# plt.grid()
# plt.show()







# import numpy as np
# import matplotlib.pyplot as plt

# def x_t(t, x0, epsilon, sigma):
#     numerator = x0 * epsilon
#     denominator = (epsilon - sigma * x0) * np.exp(-epsilon * t) + sigma * x0
#     return numerator / denominator

# # Часовий інтервал
# t = np.linspace(0, 10, 1000)

# # Параметри
# x0_values = [5, 10, 15]
# epsilon_values = [0.5, 1.0, 1.5]
# sigma_values = [0.1, 0.2, 0.3]

# # 1. Графіки для різних значень ε при фіксованих σ та x0
# sigma = 0.2
# x0 = 10

# plt.figure(figsize=(10, 6))
# for epsilon in epsilon_values:
#     x = x_t(t, x0, epsilon, sigma)
#     plt.plot(t, x, label=f'ε = {epsilon}')
# plt.title('Залежність x(t) при різних ε')
# plt.xlabel('Час t')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid()
# plt.show()

# # 2. Графіки для різних значень σ при фіксованих ε та x0
# epsilon = 1.0
# x0 = 10

# plt.figure(figsize=(10, 6))
# for sigma in sigma_values:
#     x = x_t(t, x0, epsilon, sigma)
#     plt.plot(t, x, label=f'σ = {sigma}')
# plt.title('Залежність x(t) при різних σ')
# plt.xlabel('Час t')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid()
# plt.show()

# # 3. Графіки для різних значень x0 при фіксованих ε та σ
# epsilon = 1.0
# sigma = 0.2

# plt.figure(figsize=(10, 6))
# for x0 in x0_values:
#     x = x_t(t, x0, epsilon, sigma)
#     plt.plot(t, x, label=f'x₀ = {x0}')
# plt.title('Залежність x(t) при різних x₀')
# plt.xlabel('Час t')
# plt.ylabel('Чисельність популяції x(t)')
# plt.legend()
# plt.grid()
# plt.show()


