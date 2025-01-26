# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint

# # ---------------------------------------------------------------------------------------------
# # 1
# # Параметри моделі
# R = 4
# T = 80 * 365  # Середня тривалість життя в днях (~80 років)
# n = 200       # Загальна чисельність популяції
# tau = 21      # Середня тривалість захворювання (дні)
# gamma = 1 / T
# alpha = R / (n * tau)
# beta = 1 / tau

# # Початкові умови
# Ni0 = 50
# Ns0 = 150  # n - Ni0

# # Система диференціальних рівнянь
# def model(y, t):
#     Ni, Ns = y
#     dNi_dt = alpha * Ns * Ni - beta * Ni
#     dNs_dt = gamma * (n - Ns) - alpha * Ns * Ni
#     return [dNi_dt, dNs_dt]

# # Часова сітка
# t = np.linspace(0, 365, 365)

# # Розв'язок системи
# y0 = [Ni0, Ns0]
# result = odeint(model, y0, t)

# # Побудова графіка
# plt.plot(t, result[:, 0] / n, label='$N_I / N$ (Інфіковані)')
# plt.plot(t, result[:, 1] / n, label='$N_S / N$ (Сприйнятливі)')
# plt.xlabel('Час, дні')
# plt.ylabel('Частка популяції')
# plt.title('Динаміка інфікованих та сприйнятливих (Графік 1)')
# plt.legend()
# plt.grid()
# plt.show()


# # ---------------------------------------------------------------------------------------------
# # 2
# # Параметри моделі
# R = 4
# T = 80 * 365       # Середня тривалість життя (дні)
# n = 1_000_000      # Загальна чисельність популяції
# tau = 21           # Середня тривалість захворювання (дні)
# gamma = 1 / T      # 1/дні
# beta = 1 / tau     # 1/дні
# alpha = R / (n * tau)  # 1/(особа*дні)

# # Початкові умови
# Ni0 = 10           # Початкова кількість інфікованих
# Ns0 = n - Ni0      # Початкова кількість сприйнятливих

# # Система диференціальних рівнянь
# def model(y, t):
#     Ni, Ns = y
#     dNi_dt = alpha * Ns * Ni - beta * Ni
#     dNs_dt = gamma * (n - Ns) - alpha * Ns * Ni
#     return [dNi_dt, dNs_dt]

# # Часова сітка
# t = np.linspace(0, 365 * 1, 1000)  # 1 років у днях

# # Розв'язок системи
# y0 = [Ni0, Ns0]
# result = odeint(model, y0, t)

# # Побудова графіка
# plt.figure(figsize=(12, 6))
# plt.plot(t / 365, result[:, 0] / n, label='$N_I / N$ (Інфіковані)')
# plt.plot(t / 365, result[:, 1] / n, label='$N_S / N$ (Сприйнятливі)')
# plt.xlabel('Час, роки')
# plt.ylabel('Частка популяції')
# plt.title('Динаміка інфікованих та сприйнятливих (Графік 2)')
# plt.legend()
# plt.grid()
# plt.show()



# # ---------------------------------------------------------------------------------------------
# # 3
# # Параметри моделі
# R = 4
# T = 80 * 365
# n = 200
# tau = 40  # Днів
# gamma = 1 / T
# alpha = R / (n * tau)
# beta = 1 / tau

# # Початкові умови
# Ni0 = 50
# Ns0 = 150

# # Система диференціальних рівнянь
# def model(y, t):
#     Ni, Ns = y
#     dNi_dt = alpha * Ns * Ni - beta * Ni
#     dNs_dt = gamma * (n - Ns) - alpha * Ns * Ni
#     return [dNi_dt, dNs_dt]

# # Часова сітка
# t = np.linspace(0, 365, 365)

# # Розв'язок системи
# y0 = [Ni0, Ns0]
# result = odeint(model, y0, t)

# # Побудова графіка
# plt.plot(t, result[:, 0] / n, label='$N_I / N$ (Інфіковані)')
# plt.plot(t, result[:, 1] / n, label='$N_S / N$ (Сприйнятливі)')
# plt.xlabel('Час, дні')
# plt.ylabel('Частка популяції')
# plt.title('Вплив збільшення τ на динаміку (Графік 3)')
# plt.legend()
# plt.grid()
# plt.show()


# # ---------------------------------------------------------------------------------------------
# # 4
# # Параметри моделі
# R = 4
# T = 80       # Років
# n = 200
# tau = 0.1096  # Років (~40 днів)
# gamma = 1 / T
# alpha = R / (n * tau)
# beta = 1 / tau

# # Початкові умови
# Ni0 = 50
# Ns0 = 150

# # Система диференціальних рівнянь
# def model(y, t):
#     Ni, Ns = y
#     dNi_dt = alpha * Ns * Ni - beta * Ni
#     dNs_dt = gamma * (n - Ns) - alpha * Ns * Ni
#     return [dNi_dt, dNs_dt]

# # Часова сітка
# t = np.linspace(0, 5, 500)  # 5 років

# # Розв'язок системи
# y0 = [Ni0, Ns0]
# result = odeint(model, y0, t)

# # Побудова графіка
# plt.plot(t, result[:, 0] / n, label='$N_I / N$ (Інфіковані)')
# plt.plot(t, result[:, 1] / n, label='$N_S / N$ (Сприйнятливі)')
# plt.xlabel('Час, роки')
# plt.ylabel('Частка популяції')
# plt.title('Динаміка з τ = 0.1096 років (Графік 4)')
# plt.legend()
# plt.grid()
# plt.show()


# # ---------------------------------------------------------------------------------------------
# #5-6
# # Параметри моделі
# T = 80 * 365      # Середня тривалість життя (дні)
# n = 1_000_000
# tau = 21          # Середня тривалість захворювання (дні)
# gamma = 1 / T     # 1/дні
# R = 4
# beta = 1 / tau    # 1/дні
# alpha0 = R / (n * tau)  # 1/(особа*дні)
# A = 0.1 * alpha0        # Амплітуда коливань
# omega1 = 2 * np.pi / 365  # Кутова частота (період 1 рік у днях)
# phi = 0

# # Початкові умови
# Ni0 = 1000
# Ns0 = n - Ni0

# # Система диференціальних рівнянь
# def model(y, t):
#     Ni, Ns = y
#     alpha_t = alpha0 + A * np.sin(omega1 * t + phi)
#     dNi_dt = alpha_t * Ns * Ni - beta * Ni
#     dNs_dt = -alpha_t * Ns * Ni + gamma * (n - Ns)
#     return [dNi_dt, dNs_dt]

# # Часова сітка
# t = np.linspace(0, 365 * 1, 1000)  # 1 років у днях

# # Розв'язок системи
# y0 = [Ni0, Ns0]
# result = odeint(model, y0, t)

# # Графік 5: α(t) * Ns(t) * Ni(t)
# plt.figure(figsize=(12,5))
# plt.plot(t / 365, (alpha0 + A * np.sin(omega1 * t + phi)) * result[:, 1] * result[:, 0])
# plt.xlabel('Час, роки')
# plt.ylabel('α(t) * Ns * Ni')
# plt.title('Періодично змінний коефіцієнт передачі (Графік 5)')
# plt.grid()
# plt.show()

# # Графік 6: Ni(t) та Ns(t)
# plt.figure(figsize=(12,5))
# plt.plot(t / 365, result[:, 0], label='$N_I(t)$ (Інфіковані)')
# plt.plot(t / 365, result[:, 1], label='$N_S(t)$ (Сприйнятливі)')
# plt.xlabel('Час, роки')
# plt.ylabel('Кількість осіб')
# plt.title('Динаміка Ni та Ns з періодичним α(t) (Графік 6)')
# plt.legend()
# plt.grid()
# plt.show()



# # ---------------------------------------------------------------------------------------------
# # 7-8
# # Параметри моделі
# T = 80       # Років
# n = 10000
# tau = 0.058  # Років (~21 день)
# gamma = 1 / T
# R = 4
# alpha0 = R / (n * tau)
# beta = 1 / tau
# A = 0.10
# omega1 = 2 * np.pi
# phi = 0.01

# # Функція ймовірності передачі інфекції
# def transmission_probability(t, active_start, period):
#     # Функція повертає 1, якщо t в періоді активної передачі, і 0 інакше
#     return np.heaviside(np.sin(2 * np.pi * (t - active_start) / period), 0)

# # Система диференціальних рівнянь з ймовірністю передачі
# def model(y, t):
#     Ni, Ns = y
#     alpha_t = alpha0 + A * np.sin(omega1 * t + phi)
#     transmission = alpha_t * Ns * Ni * transmission_probability(t, 0, 0.5)
#     dNi_dt = transmission - beta * Ni
#     dNs_dt = -transmission + gamma * (n - Ns)
#     return [dNi_dt, dNs_dt]

# # Початкові умови
# Ni0 = 1000
# Ns0 = n - Ni0

# # Часова сітка
# t = np.linspace(0, 5, 1000)  # 5 років

# # Розв'язок системи
# y0 = [Ni0, Ns0]
# result = odeint(model, y0, t)

# # Графік 7: α(t) * Ns(t) * Ni(t) з ймовірністю передачі
# plt.figure(figsize=(12,5))
# plt.plot(t, (alpha0 + A * np.sin(omega1 * t + phi)) * result[:, 0] * result[:, 1] * transmission_probability(t, 0, 0.5))
# plt.xlabel('Час, роки')
# plt.ylabel('α(t) * Ns * Ni * P(t)')
# plt.title('Передача інфекції з врахуванням ймовірності (Графік 7)')
# plt.grid()
# plt.show()

# # Графік 8: Ni(t) та Ns(t) з ймовірністю передачі
# plt.figure(figsize=(12,5))
# plt.plot(t, result[:, 0], label='$N_I(t)$ (Інфіковані)')
# plt.plot(t, result[:, 1], label='$N_S(t)$ (Сприйнятливі)')
# plt.xlabel('Час, роки')
# plt.ylabel('Кількість осіб')
# plt.title('Динаміка Ni та Ns з ймовірністю передачі (Графік 8)')
# plt.legend()
# plt.grid()
# plt.show()


# # ---------------------------------------------------------------------------------------------
# # 9
# # Випадкові параметри
# R = np.random.uniform(2, 5)
# T = np.random.uniform(1, 10)
# tau = np.random.uniform(0.1, 2)
# gamma = 1 / T
# beta = 1 / tau

# # Розрахунок ω та φ
# omega = np.sqrt(T / tau)
# phi = np.arctan(np.sqrt(R * (R - 2)) / (R - 1))

# # Часова сітка
# t = np.linspace(0, 50, 1000)

# # Розв'язок з додаванням шуму
# Ni = (1/3) + (1/3) * np.cos(omega * t + phi) + np.random.normal(0, 0.05, size=len(t))
# Ns = (1/3) + (2/3) * np.cos(omega * t + phi) + np.random.normal(0, 0.05, size=len(t))

# # Побудова графіка
# plt.figure(figsize=(12,5))
# plt.plot(t, Ni, label='$N_I(t)$')
# plt.plot(t, Ns, label='$N_S(t)$')
# plt.xlabel('Час')
# plt.ylabel('Частка популяції')
# plt.title('Динаміка з випадковими параметрами та шумом (Графік 9)')
# plt.legend()
# plt.grid()
# plt.show()

# # Вивід випадкових параметрів
# print(f'Випадкові параметри: R = {R:.2f}, T = {T:.2f}, τ = {tau:.2f}')






# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from scipy.interpolate import interp1d

# # Параметри
# R = 3
# T = 10 * 365 
# tau = 7 
# gamma = 1 / T
# beta = 1 / tau

# # Часова затримка (Δt)
# Delta_t = 2  # 2 дні
# n = 1_000_000
# alpha0 = R / (n * tau)

# # Початкові умови
# Ni0 = 100
# Ns0 = n - Ni0

# # Часова сітка
# t0 = 0
# t_max = 125
# t_eval = np.linspace(t0, t_max, 1000)

# def alpha(t):
#     return alpha0  # α постійне

# def history_Ni(t):
#     return Ni0

# def history_Ns(t):
#     return Ns0

# # Клас для розв'язання DDE
# class DDEModel:
#     def __init__(self):
#         self.t_vals = []
#         self.Ni_vals = []
#         self.Ns_vals = []

#     def dde_system(self, t, Y):
#         Ni_t = Y[0]  # N_I(t)
#         Ns_t = Y[1]  # N_S(t)

#         self.t_vals.append(t)
#         self.Ni_vals.append(Ni_t)
#         self.Ns_vals.append(Ns_t)

#         # Якщо t - Delta_t < t0, використовуємо початкову функцію
#         if t - Delta_t < t0:
#             Ni_delay = history_Ni(t - Delta_t)
#         else:
#             # Інтерполюємо N_I(t - Delta_t)
#             Ni_interp = interp1d(self.t_vals, self.Ni_vals, kind='linear', fill_value="extrapolate")
#             Ni_delay = Ni_interp(t - Delta_t)

#         dNi_dt = alpha(t) * Ns_t * Ni_delay - beta * Ni_t
#         dNs_dt = -alpha(t) * Ns_t * Ni_delay + gamma * (n - Ns_t)

#         return [dNi_dt, dNs_dt]

# # Створюємо екземпляр моделі
# dde_model = DDEModel()

# # Початкові умови
# Y0 = [Ni0, Ns0]

# # Розв'язання системи диференціальних рівнянь із запізненням
# sol = solve_ivp(dde_model.dde_system, [t0, t_max], Y0, t_eval=t_eval, method='RK45', atol=1e-6, rtol=1e-3)

# t = sol.t
# Ni = sol.y[0]
# Ns = sol.y[1]

# plt.figure(figsize=(12, 6))
# plt.plot(t, alpha0*Ns*Ni, label='$aN_iN_s$')
# plt.xlabel('Час, дні')
# plt.ylabel('Частка популяції')
# plt.title('(Графік 9)')
# plt.legend()
# plt.grid()
# plt.show()

# print(f'Параметри:')
# print(f'R = {R:.2f}')
# print(f'T = {T / 365:.2f} років')
# print(f'τ = {tau:.2f} днів')
# print(f'Δt = {Delta_t:.2f} днів')












# import numpy as np
# import matplotlib.pyplot as plt
# from ddeint import ddeint

# # Параметри моделі
# R = 3
# T = 10 * 365
# tau = 7
# gamma = 1 / T
# beta = 1 / tau

# # Часова затримка (Δt)
# Delta_t = 2  # дні
# n = 1_000_000
# alpha0 = R / (n * tau)

# A_inter = 0.5 * alpha0
# T_inter = 365
# omega = 2 * np.pi / T_inter

# # Початкові умови
# Ni0 = 100
# Ns0 = n - Ni0

# # Часова сітка
# t0 = 0
# t_max = 125
# t = np.linspace(t0, t_max, 1000)

# # Функція α(t)
# def alpha(t):
#     return alpha0  # α постійне

# # Початкова історія (функція історії)
# def history(t):
#     return np.array([Ni0, Ns0])

# # Система диференціальних рівнянь із запізненням
# def model(Y, t):
#     Ni_t, Ns_t = Y(t)
#     Ni_delay = Y(t - Delta_t)


#     if t > Delta_t:
#         Ni_delay = Y(t - Delta_t)
#     else:
#         Ni_delay = 0

#     # Диференціальні рівняння
#     dNi_dt = alpha(t) * Ns_t * Ni_delay - beta * Ni_t
#     dNs_dt = -alpha(t) * Ns_t * Ni_delay + gamma * (n - Ns_t)

#     return np.array([dNi_dt, dNs_dt])

# # Розв'язання системи DDE
# sol = ddeint(model, history, t)

# # Отримуємо результати
# Ni = sol[:, 0]
# Ns = sol[:, 1]

# # Обчислюємо захворюваність α * N_S * N_I
# I_t = (alpha0 + A_inter * np.sin(omega * t)) * Ns * Ni

# # Побудова графіка
# plt.figure(figsize=(12, 6))
# plt.plot(t, I_t, label='$\\alpha N_S N_I$')
# plt.xlabel('Час, дні')
# plt.ylabel('Захворюваність')
# plt.title('Залежність захворюваності від часу (Графік 9)')
# plt.legend()
# plt.grid()
# plt.show()

# # Вивід параметрів
# print(f'Параметри:')
# print(f'R = {R:.2f}')
# print(f'T = {T / 365:.2f} років')
# print(f'τ = {tau:.2f} днів')
# print(f'Δt = {Delta_t:.2f} днів')












# import numpy as np
# import matplotlib.pyplot as plt
# from ddeint import ddeint

# # Параметри моделі
# R0 = 3
# T = 10 * 365
# tau = 7
# gamma = 1 / T
# beta = 1 / tau



# # Часова затримка (Δt)
# Delta_t = 2  # дні
# n = 1_000_000
# alpha0 = R0 / (n * tau)

# A_inter = 0.5 * alpha0
# T_inter = 365
# omega = 2 * np.pi / T_inter




# # Початкові умови
# Ni0 = 100
# Ns0 = n - Ni0

# # Часова сітка
# t0 = 0
# t_max = 125
# t = np.linspace(t0, t_max, 1000)

# # Функція α(t) з коливаннями
# def alpha(t):
#     return alpha0 + A_inter * np.sin(omega * t)

# # Початкова історія (функція історії)
# def history(t):
#     return np.array([Ni0, Ns0])

# # Система диференціальних рівнянь із запізненням
# def model(Y, t):
#     Ni_t, Ns_t = Y(t)
    
#     # Отримуємо затримане значення Ni_delay
#     if t - Delta_t < 0:
#         Ni_delay = history(t - Delta_t)[0]
#     else:
#         Ni_delay = Y(t - Delta_t)[0]
    
#     # Поточне значення α(t)
#     alpha_t = alpha(t)
    
#     # Диференціальні рівняння
#     dNi_dt = alpha_t * Ns_t * Ni_delay - beta * Ni_t
#     dNs_dt = -alpha_t * Ns_t * Ni_delay + gamma * (n - Ns_t)
    
#     return np.array([dNi_dt, dNs_dt])

# # Розв'язання системи DDE
# sol = ddeint(model, history, t)

# # Отримуємо результати
# Ni = sol[:, 0]
# Ns = sol[:, 1]

# # Обчислюємо захворюваність α(t) * N_S(t) * N_I(t)
# I_t = alpha(t) * Ns * Ni


# # Побудова графіків
# plt.figure(figsize=(12, 10))

# # Графік 1: Захворюваність від часу
# plt.subplot(2, 1, 1)
# plt.plot(t, I_t, label='$\\alpha(t) N_S(t) N_I(t)$')
# plt.xlabel('Час, дні')
# plt.ylabel('Захворюваність')
# plt.title('Залежність захворюваності від часу')
# plt.legend()
# plt.grid()

# # # Графік 2: R(t) від часу
# # plt.subplot(2, 1, 2)
# # plt.plot(t, R_t, label='$R(t)$', color='orange')
# # plt.xlabel('Час, дні')
# # plt.ylabel('$R(t)$')
# # plt.title('Залежність $R(t)$ від часу')
# # plt.legend()
# # plt.grid()

# # plt.tight_layout()
# # plt.show()

# # Вивід параметрів
# print(f'Параметри:')
# print(f'R0 = {R0:.2f}')
# print(f'T = {T / 365:.2f} років')
# print(f'τ = {tau:.2f} днів')
# print(f'Δt = {Delta_t:.2f} днів')













# import numpy as np
# import matplotlib.pyplot as plt
# from ddeint import ddeint

# # Параметри моделі
# R = 3
# T = 10 * 365
# tau = 7
# gamma = 1 / T
# beta = 1 / tau

# # Часова затримка (Δt)
# Delta_t = 2  # дні
# n = 1_000_000
# alpha0 = R / (n * tau)

# A_inter = 0.5 * alpha0
# T_inter = 365
# omega = 2 * np.pi / T_inter


# # Початкові умови
# Ni0 = 100
# Ns0 = n - Ni0

# # Часова сітка
# t0 = 0
# t_max = 125
# t = np.linspace(t0, t_max, 1000)

# a_t = alpha0 + A_inter * np.sin(omega * t) 

# # Початкова історія (функція історії)
# def history(t):
#     return np.array([Ni0, Ns0])

# # Система диференціальних рівнянь із запізненням
# def model(Y, t):
#     Ni_t, Ns_t = Y(t)

#     if t > Delta_t:
#         Ni_delay = history(t - Delta_t)
#     else:
#         Ni_delay = 0

#     # Диференціальні рівняння
#     dNi_dt = a_t * Ns_t * Ni_delay - beta * Ni_t
#     dNs_dt = -a_t * Ns_t * Ni_delay + gamma * (n - Ns_t)

#     return np.array([dNi_dt, dNs_dt])

# # Розв'язання системи DDE
# sol = ddeint(model, history, t)

# # Отримуємо результати
# Ni = sol[:, 0]
# Ns = sol[:, 1]

# # Обчислюємо захворюваність α * N_S * N_I
# I_t = a_t * Ns * Ni

# # Побудова графіка
# plt.figure(figsize=(12, 6))
# plt.plot(t, I_t, label='$\\alpha N_S N_I$')
# plt.xlabel('Час, дні')
# plt.ylabel('Захворюваність')
# plt.title('Залежність захворюваності від часу (Графік 9)')
# plt.legend()
# plt.grid()
# plt.show()

# # Вивід параметрів
# print(f'Параметри:')
# print(f'R = {R:.2f}')
# print(f'T = {T / 365:.2f} років')
# print(f'τ = {tau:.2f} днів')
# print(f'Δt = {Delta_t:.2f} днів')









# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from scipy.interpolate import interp1d

# # Параметри моделі
# R = 3
# T = 10 * 365
# tau = 7
# gamma = 1 / T
# beta = 1 / tau

# # Часова затримка (Δt)
# Delta_t = 10  # дні
# n = 1_000_000
# alpha0 = R / (n * tau)



# A_inter = 0.5 * alpha0
# T_inter = 365
# omega = 2 * np.pi / T_inter


# # Початкові умови
# Ni0 = 100
# Ns0 = n - Ni0

# # Часова сітка
# t0 = 0
# t_max = 125
# dt = 0.01
# t_points = np.arange(t0, t_max + dt, dt)

# # Початкова історія
# def history(t):
#     return [Ni0 * (1 + 0.1 * np.sin(2 * np.pi * t / 365)), Ns0]

# # Ініціалізація масивів для зберігання результатів
# Ni_vals = []
# Ns_vals = []
# t_vals = []

# # Функція для моделі
# def dde_system(t, y):
#     Ni_t = y[0]
#     Ns_t = y[1]

#     t_delay = t - Delta_t

#     if t_delay < t0:
#         Ni_delay = Ni0
#     else:
#         Ni_delay = Ni_interp(t_delay)

#     # alpha_t = alpha0 + A_inter * np.sin(omega * t)
#     alpha_t = alpha0 + A_inter * np.sin(omega * t + np.pi/4)


#     dNi_dt = alpha_t * Ns_t * Ni_delay - beta * Ni_t
#     dNs_dt = -alpha_t * Ns_t * Ni_delay + gamma * (n - Ns_t)

#     return [dNi_dt, dNs_dt]

# # Початкові умови
# y0 = [Ni0, Ns0]

# # Інтерполяція для затримки
# Ni_interp = interp1d([t0], [Ni0], fill_value='extrapolate')

# for i in range(len(t_points) - 1):
#     t_span = [t_points[i], t_points[i+1]]

#     # sol = solve_ivp(dde_system, t_span, y0, method='RK45', t_eval=[t_points[i+1]])
#     sol = solve_ivp(dde_system, t_span, y0, method='Radau', t_eval=[t_points[i+1]])

#     Ni_t = sol.y[0][-1]
#     Ns_t = sol.y[1][-1]

#     t_vals.append(t_points[i+1])
#     Ni_vals.append(Ni_t)
#     Ns_vals.append(Ns_t)

#     # Оновлюємо інтерполятор
#     Ni_interp = interp1d(t_vals, Ni_vals, fill_value='extrapolate')

#     # Оновлюємо початкові умови для наступного кроку
#     y0 = [Ni_t, Ns_t]

# # Перетворюємо результати в numpy масиви
# t_vals = np.array(t_vals)
# Ni_vals = np.array(Ni_vals)
# Ns_vals = np.array(Ns_vals)

# # Обчислюємо захворюваність α * N_S * N_I
# alpha_t = alpha0 + A_inter * np.sin(omega * t_vals)

# I_t = alpha_t * Ns_vals * Ni_vals

# # Побудова графіка
# plt.figure(figsize=(12, 6))
# plt.plot(t_vals, I_t, label='$\\alpha(t) N_S(t) N_I(t)$')
# plt.xlabel('Час, дні')
# plt.ylabel('Захворюваність')
# plt.title('Залежність захворюваності від часу (Графік 9)')
# plt.legend()
# plt.grid()
# plt.show()

# # Вивід параметрів
# print(f'Параметри:')
# print(f'R = {R:.2f}')
# print(f'T = {T / 365:.2f} років')
# print(f'τ = {tau:.2f} днів')
# print(f'Δt = {Delta_t:.2f} днів')












# import numpy as np
# import matplotlib.pyplot as plt
# from ddeint import ddeint

# # Параметри моделі
# R = 3
# T = 10 * 365
# tau = 7
# gamma = 1 / T
# beta = 1 / tau


# # Часова затримка (Δt)
# Delta_t = 2  # дні
# n = 1_000_000
# alpha0 = R / (n * tau)

# A_inter = 0.5 * alpha0
# T_inter = 365
# omega = 2 * np.pi / T_inter


# # Початкові умови
# Ni0 = 100
# Ns0 = n - Ni0

# # Часова сітка
# t0 = 0
# t_max = 125
# t = np.linspace(t0, t_max, 1000)


# # Початкова історія (функція історії)
# def history(t):
#     return np.array([Ni0, Ns0])

# # Система диференціальних рівнянь із запізненням
# def model(Y, t):
#     Ns_t = Y(t)

#     if t > Delta_t:
#         Ni_t, _ = Y(t - Delta_t)
#     else:
#         Ni_t = 0
        
#     Ni_delay = Y(t - Delta_t)

#     alpha_t = alpha0 + A_inter * np.sin(omega * t+)

#     # Диференціальні рівняння
#     dNi_dt = alpha_t * Ns_t * Ni_delay - beta * Ni_t
#     dNs_dt = -alpha_t * Ns_t * Ni_delay + gamma * (n - Ns_t)

#     return np.array([dNi_dt, dNs_dt])

# # Розв'язання системи DDE
# sol = ddeint(model, history, t)

# # Отримуємо результати
# Ni = sol[:, 0]
# Ns = sol[:, 1]

# # Обчислюємо захворюваність α * N_S * N_I
# I_t = alpha0 * Ns * Ni

# # Побудова графіка
# plt.figure(figsize=(12, 6))
# plt.plot(t, I_t, label='$\\alpha N_S N_I$')
# plt.xlabel('Час, дні')
# plt.ylabel('Захворюваність')
# plt.title('Залежність захворюваності від часу (Графік 9)')
# plt.legend()
# plt.grid()
# plt.show()

# # Вивід параметрів
# print(f'Параметри:')
# print(f'R = {R:.2f}')
# print(f'T = {T / 365:.2f} років')
# print(f'τ = {tau:.2f} днів')
# print(f'Δt = {Delta_t:.2f} днів')






# # НАЙПРОСТІША МОДЕЛЬ

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp


# # Початкові умови
# N = 1000  # загальна популяція
# R = 4  # середнє число заражених від однієї людини
# tau = 14  # тривалість захворювання (в днях)
# T = 70 * 365  # середня тривалість життя (в днях)

# alpha = R / (tau * N)  # коефіцієнт зараження (поправка до формули)
# beta = 1 / tau  # коефіцієнт видужання
# gamma = 1 / T  # коефіцієнт смертності

# # Уточнення початкових умов
# N_S0 = N * (1 / R + 0.01)  # початкова частка сприйнятливих
# N_I0 = 1  # початкова кількість інфікованих
# y0 = [N_S0, N_I0]


# # Часова шкала
# t_span = (0, 20*365)  # 10 років в днях
# t_eval = np.linspace(t_span[0], t_span[1], 1000)



# def epidemic_model(t, y):
#     N_S, N_I = y
#     N_R = N - N_S - N_I
#     dN_S_dt = -alpha * N_S * N_I + gamma * N_R
#     dN_I_dt = alpha * N_S * N_I - beta * N_I - gamma * N_I
#     return [dN_S_dt, dN_I_dt]



# # Чисельне рішення із покращеним методом
# solution = solve_ivp(epidemic_model, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

# # Результати
# t = solution.t
# N_S, N_I = solution.y

# # Побудова нового графіка
# plt.figure(figsize=(10, 6))
# plt.plot(t / 365, N_I / N, label='Інфіковані', linewidth=2, color='red')
# plt.title("Залежність відносного числа інфікованих від часу (уточнено)", fontsize=14)
# plt.xlabel("Час (роки)", fontsize=12)
# plt.ylabel("Частка інфікованих", fontsize=12)
# plt.grid()
# plt.legend()
# plt.show()




# from pylab import cos, linspace, subplots
# from ddeint import ddeint

# We solve the following system:
# Y(t) = 1 for t < 0
# dY/dt = -Y(t - 3cos(t)**2) for t > 0




# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from ddeint import ddeint

# alpha = 0.002
# beta = 0.1
# gamma = 0.01
# N_total = 1000

# N_I0 = 100
# N_S0 = N_total - N_I0

# def epidemic_model(t, y):
#     N_S, N_I = y

#     dN_I = alpha * N_I * N_S - beta * N_I
#     dN_S = -alpha * N_I * N_S + gamma * (N_total - N_S - N_I)
#     return [dN_S, dN_I]

# t_span = (0, 50)
# t_eval = np.linspace(t_span[0], t_span[1], 500)

# y0 = [N_S0, N_I0]


# solution = ddeint(epidemic_model, y0, t_span)


# t = solution.t
# N_S = solution.y[0]
# N_I = solution.y[1]
# aN_sN_i = alpha * N_S * N_I

# # Plotting the results
# plt.figure(figsize=(10, 6))

# plt.plot(t, N_S, label='Susceptible (N_S)')
# plt.plot(t, N_I, label='Infected (N_I)')
# plt.plot(t, aN_sN_i, label='aN_S N_I (infection rate)', linestyle='--')

# plt.title('Epidemic Model Dynamics')
# plt.xlabel('Time')
# plt.ylabel('Population / Rate')
# plt.legend()
# plt.grid()
# plt.show()








# import numpy as np
# import matplotlib.pyplot as plt
# from ddeint import ddeint

# # Параметри моделі
# alpha = 0.002  # Коефіцієнт інфікування
# beta = 0.1     # Коефіцієнт одужання
# gamma = 0.01   # Коефіцієнт поповнення вразливих
# N_total = 1000 # Загальна популяція

# N_I0 = 100      # Початкове число інфікованих
# N_S0 = N_total - N_I0  # Початкове число вразливих

# tau = 5  # Затримка (наприклад, інкубаційний період у днях)

# # Історична функція: стан системи до початку моделювання
# def g(t):
#     return [N_S0, N_I0]

# # Модель із затримкою
# def epidemic_model(Y, t):
#     # Поточні значення
#     N_S, N_I = Y(t)
#     # Значення із затримкою
#     N_S_tau, N_I_tau = Y(t - tau)

#     # Диференціальні рівняння
#     dN_I = alpha * N_I_tau * N_S - beta * N_I
#     dN_S = -alpha * N_I_tau * N_S + gamma * (N_total - N_S - N_I)
#     return [dN_S, dN_I]

# # Часова шкала
# t_span = np.linspace(0, 30, 1000)

# # Розв'язання DDE
# solution = ddeint(epidemic_model, g, t_span)

# # Результати
# N_S = solution[:, 0]
# N_I = solution[:, 1]
# aN_sN_i = alpha * N_S * N_I

# # Побудова графіків
# plt.figure(figsize=(10, 6))

# plt.plot(t_span, N_S, label='Susceptible (N_S)')
# plt.plot(t_span, N_I, label='Infected (N_I)')
# plt.plot(t_span, aN_sN_i, label='aN_S N_I (infection rate)', linestyle='--')

# plt.title('Epidemic Model Dynamics with Delay')
# plt.xlabel('Time')
# plt.ylabel('Population / Rate')
# plt.legend()
# plt.grid()
# plt.show()










# import numpy as np
# import matplotlib.pyplot as plt
# from ddeint import ddeint

# # Параметри моделі
# alpha = 0.002
# beta = 0.1
# gamma = 0.01
# N_total = 1000

# N_I0 = 100
# N_S0 = N_total - N_I0

# tau = 5

# def g(t):
#     return [N_S0, N_I0]

# # Модель із затримкою
# def epidemic_model(Y, t):
#     N_S, N_I = Y(t)
#     N_S_tau, N_I_tau = Y(t - tau)

#     # Диф рівняння
#     dN_I = alpha * N_I_tau * N_S - beta * N_I
#     dN_S = -alpha * N_I_tau * N_S + gamma * (N_total - N_S - N_I)
#     return [dN_S, dN_I]

# t_span = np.linspace(0, 30, 30)

# # Розв'язання DDE
# solution = ddeint(epidemic_model, g, t_span)

# N_S = solution[:, 0]
# N_I = solution[:, 1]
# aN_sN_i = alpha * N_S * N_I

# plt.figure(figsize=(10, 6))

# plt.plot(t_span, N_S, label='Susceptible (N_S)')
# plt.plot(t_span, N_I, label='Infected (N_I)')
# plt.plot(t_span, aN_sN_i, label='aN_S N_I (infection rate)', linestyle='--')

# plt.title('E.M.D. with Delay')
# plt.xlabel('t')
# plt.ylabel('Population / Rate')
# plt.legend()
# plt.grid()
# plt.show()








# import numpy as np
# import matplotlib.pyplot as plt
# from ddeint import ddeint

# # Параметри моделі
# alpha_0 = 0.002  # Базове значення α
# A = 0.001        # Амплітуда
# T = 1            # Період у роках
# omega = 2 * np.pi / T  # Кутова частота
# phi = 0          # Початкова фаза

# beta = 0.1       # Коефіцієнт одужання
# gamma = 0.01     # Коефіцієнт поповнення вразливих
# N_total = 1000   # Загальна популяція

# N_I0 = 100       # Початкове число інфікованих
# N_S0 = N_total - N_I0  # Початкове число вразливих

# tau = 5          # Затримка

# # Історична функція
# def g(t):
#     return [N_S0, N_I0]

# # Періодична функція α(t)
# def periodic_alpha(t):
#     return alpha_0 + A * np.sin(omega * t + phi)

# # Модель із затримкою
# def epidemic_model(Y, t):
#     N_S, N_I = Y(t)
#     N_S_tau, N_I_tau = Y(t - tau)
    
#     # Періодичний коефіцієнт α(t)
#     alpha_t = periodic_alpha(t)

#     # Диференціальні рівняння
#     dN_S = -alpha_t * N_I_tau * N_S + gamma * (N_total - N_S - N_I)
#     dN_I = alpha_t * N_I_tau * N_S - beta * N_I
#     return [dN_S, dN_I]

# # Часова шкала
# t_span = np.linspace(0, 30, 500)

# # Розв'язання DDE
# solution = ddeint(epidemic_model, g, t_span)

# # Розпакування результатів
# N_S = solution[:, 0]
# N_I = solution[:, 1]

# # Обчислення aN_S N_I у часі
# alpha_values = periodic_alpha(t_span)
# aN_sN_i = alpha_values * N_S * N_I

# # Побудова графіків
# plt.figure(figsize=(12, 8))

# plt.plot(t_span, N_S, label='Вразливі (N_S)')
# plt.plot(t_span, N_I, label='Інфіковані (N_I)')
# plt.plot(t_span, aN_sN_i, label='aN_S N_I (темп інфікування)', linestyle='--')

# plt.title('Динаміка епідеміологічної моделі із періодичною α(t)')
# plt.xlabel('Час (роки)')
# plt.ylabel('Популяція / Темп інфікування')
# plt.legend()
# plt.grid()
# plt.show()







# import numpy as np
# import matplotlib.pyplot as plt
# from ddeint import ddeint

# # Параметри моделі
# alpha_0 = 0.002  # Базове значення α
# A = 0.001        # Амплітуда
# T = 1            # Період у роках
# omega = 2 * np.pi / T  # Кутова частота
# phi = 0          # Початкова фаза

# beta = 0.1       # Коефіцієнт одужання
# gamma = 0.01     # Коефіцієнт поповнення вразливих
# N_total = 1000   # Загальна популяція

# N_I0 = 100       # Початкове число інфікованих
# N_S0 = N_total - N_I0  # Початкове число вразливих

# tau = 5          # Затримка

# # Історична функція
# def g(t):
#     return [N_S0, N_I0]

# # Періодична функція α(t)
# def periodic_alpha(t):
#     return alpha_0 + A * np.sin(omega * t + phi)

# # Модель із затримкою
# def epidemic_model(Y, t):
#     N_S, N_I = Y(t)
#     N_S_tau, N_I_tau = Y(t - tau)
    
#     # Періодична α(t) входить до системи рівнянь
#     alpha_t = periodic_alpha(t)

#     # Диференціальні рівняння із використанням α(t)
#     dN_S = -alpha_t * N_I_tau * N_S + gamma * (N_total - N_S - N_I)
#     dN_I = alpha_t * N_I_tau * N_S - beta * N_I
#     return [dN_S, dN_I]

# # Часова шкала
# t_span = np.linspace(0, 30, 500)

# # Розв'язання DDE
# solution = ddeint(epidemic_model, g, t_span)

# # Розпакування результатів
# N_S = solution[:, 0]
# N_I = solution[:, 1]

# # aN_sN_i тепер залежить від результатів системи рівнянь
# aN_sN_i = periodic_alpha(t_span) * N_S * N_I

# # Побудова графіків
# plt.figure(figsize=(12, 8))

# plt.plot(t_span, N_S, label='Susceptible (N_S)')
# plt.plot(t_span, N_I, label='Infected (N_I)')
# plt.plot(t_span, aN_sN_i, label='aN_S N_I (infection rate)', linestyle='--')


# plt.title('E.M.D. with Delay')
# plt.xlabel('t')
# plt.ylabel('Population / Rate')
# plt.legend()
# plt.grid()
# plt.show()








import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

alpha_0 = 0.02 # збільшив
A = 0.02 # збільшив
T = 3 # збільшив
omega = 2 * np.pi / T
phi = 0

beta = 0.4 # збільшив
gamma = 0.06 # збільшив
N_total = 100

N_I0 = 100
N_S0 = N_total - N_I0

tau = 5 

def g(t):
    return [N_S0, N_I0]

def epidemic_model(Y, t):
    N_S, N_I = Y(t)
    N_S_tau, N_I_tau = Y(t - tau)
    
    alpha_t = alpha_0 + A * np.sin(omega * t + phi)

    dN_S = -alpha_t * N_I_tau * N_S + gamma * (N_total - N_S - N_I)
    dN_I = alpha_t * N_I_tau * N_S - beta * N_I
    return [dN_S, dN_I]

t_span = np.linspace(0, 100, 300)

# Розв'язання DDE
solution = ddeint(epidemic_model, g, t_span)

N_S = solution[:, 0]
N_I = solution[:, 1]

aN_sN_i = (alpha_0 + A * np.sin(omega * t_span + phi)) * N_S * N_I

# Побудова графіків
plt.figure(figsize=(12, 8))

plt.plot(t_span, N_S, label='Susceptible (N_S)')
plt.plot(t_span, N_I, label='Infected (N_I)')
plt.plot(t_span, aN_sN_i, label='aN_S N_I (infection rate)', linestyle='--')

plt.title('E.M.D. with Increased Coefficients')
plt.xlabel('t')
plt.ylabel('Population / Rate')
plt.legend()
plt.grid()
plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
# from scipy.interpolate import interp1d

# # Random parameters
# R = np.random.uniform(2, 5)
# T = np.random.uniform(1 * 365, 10 * 365)  # From 1 to 10 years in days
# tau = np.random.uniform(7, 21)  # From 7 to 21 days
# gamma = 1 / T
# beta = 1 / tau

# # Time delay (Δt)
# Delta_t = np.random.uniform(1, 14)  # From 1 to 14 days

# # Total population
# n = 1_000_000

# # Compute alpha
# alpha0 = R / (n * tau)

# # Initial conditions
# Ni0 = 100
# Ns0 = n - Ni0

# # Time grid
# t0 = 0
# t_max = 200
# dt = 0.1  # Time step
# t_points = np.arange(t0, t_max + dt, dt)

# # Initial history on interval [t0 - Delta_t, t0]
# def history(t):
#     return [Ni0, Ns0]

# # Initialize arrays to store results
# Ni_vals = []
# Ns_vals = []
# t_vals = []

# # Function for the model
# def dde_system(y, t, interpolators):
#     Ni_t = y[0]
#     Ns_t = y[1]

#     # Compute t_delay
#     t_delay = t - Delta_t

#     # If t_delay less than initial time, use history
#     if t_delay <= t0:
#         Ni_delay, Ns_delay = history(t_delay)
#     else:
#         # Use interpolation to get delayed values
#         Ni_delay = interpolators['Ni'](t_delay)
#         Ns_delay = interpolators['Ns'](t_delay)

#     # System of equations
#     dNi_dt = alpha0 * Ns_t * Ni_delay - beta * Ni_t
#     dNs_dt = -alpha0 * Ns_delay * Ni_delay + gamma * (n - Ns_t)

#     return [dNi_dt, dNs_dt]

# # Initial conditions for integration
# y0 = [Ni0, Ns0]

# # Adjust t_eval_initial to have at least two points
# t_eval_initial = np.arange(t0, t0 + Delta_t + dt, dt)
# if len(t_eval_initial) < 2:
#     t_eval_initial = np.array([t0, t0 + dt])

# # Initial integration over [t0, t0 + Delta_t]
# sol_initial = odeint(lambda y, t: dde_system(y, t, {'Ni': lambda x: Ni0, 'Ns': lambda x: Ns0}), y0, t_eval_initial)

# # Store results
# t_vals.extend(t_eval_initial)
# Ni_vals.extend(sol_initial[:, 0])
# Ns_vals.extend(sol_initial[:, 1])

# # Integration over next intervals
# for i in range(len(t_eval_initial), len(t_points)):
#     t_span = [t_points[i-1], t_points[i]]

#     # Create interpolators for Ni and Ns
#     if len(t_vals) < 2:
#         # Ensure there are enough points for interpolation
#         Ni_interp = lambda x: Ni_vals[-1]
#         Ns_interp = lambda x: Ns_vals[-1]
#     else:
#         Ni_interp = interp1d(t_vals, Ni_vals, kind='linear', fill_value="extrapolate")
#         Ns_interp = interp1d(t_vals, Ns_vals, kind='linear', fill_value="extrapolate")
#     interpolators = {'Ni': Ni_interp, 'Ns': Ns_interp}

#     # Initial conditions for this step
#     y0_step = [Ni_vals[-1], Ns_vals[-1]]

#     # Integrate over one step
#     sol = odeint(dde_system, y0_step, t_span, args=(interpolators,))

#     # Store results
#     t_vals.append(t_span[1])
#     Ni_vals.append(sol[-1, 0])
#     Ns_vals.append(sol[-1, 1])

# # Convert results to numpy arrays
# t_vals = np.array(t_vals)
# Ni_vals = np.array(Ni_vals)
# Ns_vals = np.array(Ns_vals)

# # Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(t_vals, Ni_vals / n, label='$N_I(t) / N$ (Інфіковані)')
# plt.plot(t_vals, Ns_vals / n, label='$N_S(t) / N$ (Сприйнятливі)')
# plt.xlabel('Час, дні')
# plt.ylabel('Частка популяції')
# plt.title('Динаміка з випадковими параметрами та часовою затримкою (Графік 9)')
# plt.legend()
# plt.grid()
# plt.show()

# # Output random parameters
# print(f'Випадкові параметри:')
# print(f'R = {R:.2f}')
# print(f'T = {T / 365:.2f} років')
# print(f'τ = {tau:.2f} днів')
# print(f'Δt = {Delta_t:.2f} днів')








# # ---------------------------------------------------------------------------------------------
# # 10
# # Тривіальний випадок
# R_trivial = 0
# T = 1
# tau = 1
# gamma = 1 / T
# beta = 1 / tau

# # Нетривіальний випадок
# R_nontrivial = np.random.uniform(0, 1)
# omega = np.sqrt(T / tau)
# phi = 0

# # Часова сітка
# t = np.linspace(0, 50, 1000)

# # Розв'язок для тривіального випадку
# Ni_trivial = np.zeros_like(t)
# Ns_trivial = np.zeros_like(t)

# # Розв'язок для нетривіального випадку
# Ni_nontrivial = (1/3) + (2/3) * np.cos(np.sqrt(R_nontrivial) * t + phi)
# Ns_nontrivial = (1/3) + (1/3) * np.cos(np.sqrt(R_nontrivial) * t + phi)

# # Відхилення від нульового стаціонарного розв'язку
# N_v = (1/3) + np.cos(np.sqrt(R_nontrivial) * t)
# N_w = (1/3) + np.cos(np.sqrt(R_trivial) * t)

# # Побудова графіків
# plt.figure(figsize=(12, 8))

# # Тривіальний випадок
# plt.subplot(2, 2, 1)
# plt.plot(t, Ni_trivial, label='$N_I(t)$')
# plt.plot(t, Ns_trivial, label='$N_S(t)$')
# plt.title('Тривіальний розв\'язок: R = 0')
# plt.legend()
# plt.grid()

# # Нетривіальний випадок
# plt.subplot(2, 2, 2)
# plt.plot(t, Ni_nontrivial, label='$N_I(t)$')
# plt.plot(t, Ns_nontrivial, label='$N_S(t)$')
# plt.title(f'Нетривіальний розв\'язок: R = {R_nontrivial:.2f}')
# plt.legend()
# plt.grid()

# # Відхилення від нульового стаціонарного розв'язку (R_nontrivial)
# plt.subplot(2, 2, 3)
# plt.plot(t, N_v, label='$N_v(t)$')
# plt.title('Відхилення від нульового розв\'язку (R_nontrivial)')
# plt.legend()
# plt.grid()

# # Відхилення від нульового стаціонарного розв'язку (R_trivial)
# plt.subplot(2, 2, 4)
# plt.plot(t, N_w, label='$N_w(t)$')
# plt.title('Відхилення від нульового розв\'язку (R_trivial)')
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()






























