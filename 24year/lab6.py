import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Визначення диференціального рівняння
def dPdt(P, t, W, C):
    return -W * C * P

# Параметри
W = 1      # Гідродинамічний опір (мм рт. ст.·с/мл)
C = 1.2    # Податливість судини (мл/мм рт. ст.)
P0 = 120   # Початковий тиск (мм рт. ст.)

# Часові точки
t = np.linspace(0, 5, 100)

# Розв'язок ОДУ
P = odeint(dPdt, P0, t, args=(W, C)).flatten()

# Побудова графіка
plt.figure(figsize=(8, 6))
plt.plot(t, P, label='P(t)')
plt.xlabel('Час (с)')
plt.ylabel('Тиск (мм рт. ст.)')
plt.title('Зміна тиску P(t) від часу')
plt.legend()
plt.grid(True)
plt.show()


# Визначення системи диференціальних рівнянь для Фази 1
def system_phase1(y, t, W, C, P_out, a, b):
    P, Q = y
    dP_dt = (P / (W * C)) - (Q / C) - (P_out / (W * C))
    dQ_dt = -a * t**2 + b * t
    return [dP_dt, dQ_dt]

# Параметри
W = 1          # Гідродинамічний опір (мм рт. ст.·с/мл)
C = 1.2        # Податливість судини (мл/мм рт. ст.)
P_out = 0      # Вихідний тиск (мм рт. ст.)
Q_max = 500    # Максимальна об'ємна швидкість притоку (мл/с)
t0 = 0.12      # Часовий параметр (с)

# Коефіцієнти для Q_c(t)
a = Q_max / t0**2
b = (2 * Q_max) / t0

# Початкові умови: [Тиск, Об'ємна швидкість]
initial_conditions = [60, 0]

# Часові точки для Фази 1
t_phase1 = np.linspace(0, 0.8, 100)

# Розв'язок системи ОДУ для Фази 1
solution_phase1 = odeint(system_phase1, initial_conditions, t_phase1, args=(W, C, P_out, a, b))

# Побудова графіка
plt.figure(figsize=(8, 6))
plt.plot(t_phase1, solution_phase1[:, 0], label='Тиск P(t)')
plt.plot(t_phase1, solution_phase1[:, 1], label='Об\'ємна швидкість Q(t)')
plt.xlabel('Час (с)')
plt.ylabel('Тиск (мм рт. ст.) та Об\'ємна швидкість (мл/с)')
plt.title('Фаза 1: Зміна тиску та об\'ємної швидкості від часу')
plt.legend()
plt.grid(True)
plt.show()


# Визначення системи диференціальних рівнянь для Фази 1 та Фази 2
def system(y, t, W, C, P_out, a, b, phase):
    P, Q = y
    if phase == 1:
        dP_dt = (P / (W * C)) - (Q / C) - (P_out / (W * C))
        dQ_dt = -a * t**2 + b * t
    elif phase == 2:
        dP_dt = (P / (W * C)) - (Q / C) - (P_out / (W * C))
        dQ_dt = 0  # Q = 0 у Фазі 2
    return [dP_dt, dQ_dt]

# Параметри
W = 1          # Гідродинамічний опір (мм рт. ст.·с/мл)
C = 1.2        # Податливість судини (мл/мм рт. ст.)
P_out = 0      # Вихідний тиск (мм рт. ст.)
Q_max = 500    # Максимальна об'ємна швидкість притоку (мл/с)
t0 = 0.12      # Часовий параметр (с)

# Коефіцієнти для Q_c(t)
a = Q_max / t0**2
b = (2 * Q_max) / t0

# Початкові умови для Фази 1
initial_conditions_phase1 = [60, 0]

# Часові точки для Фази 1
t_phase1 = np.linspace(0, 0.8, 100)

# Розв'язок системи ОДУ для Фази 1
solution_phase1 = odeint(system, initial_conditions_phase1, t_phase1, args=(W, C, P_out, a, b, 1))

# Початкові умови для Фази 2 (кінцеві значення Фази 1)
P1_end = solution_phase1[-1, 0]
initial_conditions_phase2 = [P1_end, 0]

# Часові точки для Фази 2
t_phase2 = np.linspace(0.8, 1.6, 100)

# Розв'язок системи ОДУ для Фази 2
solution_phase2 = odeint(system, initial_conditions_phase2, t_phase2, args=(W, C, P_out, a, b, 2))

# Побудова графіків
plt.figure(figsize=(14, 6))

# Графік для Фази 1
plt.subplot(1, 2, 1)
plt.plot(t_phase1, solution_phase1[:, 0], label='Тиск P(t)')
plt.plot(t_phase1, solution_phase1[:, 1], label='Об\'ємна швидкість Q(t)')
plt.xlabel('Час (с)')
plt.ylabel('Тиск (мм рт. ст.) та Об\'ємна швидкість (мл/с)')
plt.title('Фаза 1: Приплив крові в аорту')
plt.legend()
plt.grid(True)
plt.ylim(-15000, max(solution_phase1[:, 1]) + 1000)

# Графік для Фази 2
plt.subplot(1, 2, 2)
plt.plot(t_phase2, solution_phase2[:, 0], label='Тиск P(t)')
plt.plot(t_phase2, solution_phase2[:, 1], label='Об\'ємна швидкість Q(t)')
plt.xlabel('Час (с)')
plt.ylabel('Тиск (мм рт. ст.) та Об\'ємна швидкість (мл/с)')
plt.title('Фаза 2: Відтік крові в мікросудини')
plt.legend()
plt.grid(True)
plt.ylim(-15000, max(solution_phase2[:, 1]) + 1000)

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, dsolve, Eq, exp
from sympy.utilities.lambdify import lambdify

# Оголошення символьних змінних
t = symbols('t')
P = Function('P')

# Параметри (числові значення)
W = 1       # Гідродинамічний опір (мм рт. ст.·с/мл)
C = 1.2     # Податливість судини (мл/мм рт. ст.)
P0 = 120    # Початковий тиск при t=0 (мм рт. ст.)

# Визначення диференціального рівняння
# dP/dt + P / (W * C) = 0
equation = Eq(P(t).diff(t) + P(t) / (W * C), 0)

# Розв'язання рівняння з початковою умовою P(0) = P0
solution = dsolve(equation, P(t), ics={P(0): P0})

# Виведення розв'язку
print(f"Розв'язок рівняння: {solution}")

# Отримання виразу для P(t)
P_expr = solution.rhs

# Створення числової функції з символьного виразу
P_func = lambdify(t, P_expr, 'numpy')

# Генерація значень для графіка
t_values = np.linspace(0, 2, 100)
P_values = P_func(t_values)

# Побудова графіка
plt.figure(figsize=(8, 6))
plt.plot(t_values, P_values, label=f'P(t) = {P_expr}')
plt.xlabel('Час t (с)')
plt.ylabel('Тиск P(t) (мм рт. ст.)')
plt.title('Залежність тиску P(t) від часу')
plt.legend()
plt.grid(True)
plt.show()



# Параметри
W = 1          # Гідродинамічний опір (мм рт. ст.·с/мл)
C_values = [1.2, 1.7, 2.2]  # Різні значення податливості
PO = 105.5     # Початковий тиск (мм рт. ст.)

# Визначення системи диференціальних рівнянь
def system(P, t, W, C_values):
    return [-W * C * P_i for C, P_i in zip(C_values, P)]

# Початкові умови
P0 = [PO] * len(C_values)

# Часові точки
t = np.linspace(0, 2, 100)

# Розв'язок системи ОДУ
sol = odeint(system, P0, t, args=(W, C_values))

# Побудова графіка
plt.figure(figsize=(8, 6))
for i, C in enumerate(C_values):
    plt.plot(t, sol[:, i], label=f'C={C}')

plt.xlabel('Час (с)')
plt.ylabel('Тиск P(t) (мм рт. ст.)')
plt.title('Зміна тиску P(t) для різних значень C')
plt.legend()
plt.grid(True)
plt.show()





# Параметри
W_values = [1.0, 1.5, 2.0]  # Різні значення W
C = 1.2                    # Податливість (мл/мм рт. ст.)
PO = 120                 # Початковий тиск (мм рт. ст.)

# Визначення системи диференціальних рівнянь
def system(P, t, W, C_factors):
    return [-W * factor * C * P_i for factor, P_i in zip(C_factors, P)]

# Початкові умови
C_factors_list = [
    [1.0, 1.0, 1.0],  # Для W=1.0
    [1.5, 1.5, 1.5],  # Для W=1.5
    [2.0, 2.0, 2.0],  # Для W=2.0
]
C_factors_list = [[1.0, 1.5, 2.0]] * len(W_values)  # Множення відповідних факторів

P0 = [PO, PO, PO]

# Часові точки
t = np.linspace(0, 2, 100)

# Розв'язок системи ОДУ та побудова графіків
plt.figure(figsize=(8, 6))
for i, W in enumerate(W_values):
    C_factors = [1.2, 1.5, 2.0]  # Припущення: різні C-фактори
    sol = odeint(system, P0, t, args=(W, C_factors))
    plt.plot(t, sol[:, 0], label=f'W={W}, C=1.2')
    plt.plot(t, sol[:, 1], label=f'W={W}, C=1.5')
    plt.plot(t, sol[:, 2], label=f'W={W}, C=2.0')

plt.xlabel('Час (с)')
plt.ylabel('Тиск P(t) (мм рт. ст.)')
plt.title('Зміна тиску P(t) для різних значень W та C')
plt.legend()
plt.grid(True)
plt.show()




# Параметри
W_values = [1.0, 0.5, 0.1]  # Різні значення W
C = 1.2                    # Податливість (мл/мм рт. ст.)
PO = 120                # Початковий тиск (мм рт. ст.)

# Визначення системи диференціальних рівнянь
def system(P, t, W, C_factors):
    return [-W * factor * C * P_i for factor, P_i in zip(C_factors, P)]

# Початкові умови
C_factors_list = [
    [1.0, 0.5, 0.1],  # Для W=1.0
    [1.0, 0.5, 0.1],  # Для W=0.5
    [1.0, 0.5, 0.1],  # Для W=0.1
]
P0 = [PO, PO, PO]

# Часові точки
t = np.linspace(0, 2, 100)

# Розв'язок системи ОДУ та побудова графіків
plt.figure(figsize=(8, 6))
for W, C_factors in zip(W_values, C_factors_list):
    sol = odeint(system, P0, t, args=(W, C_factors))
    plt.plot(t, sol[:, 0], label=f'W={W}, C=1.0')
    plt.plot(t, sol[:, 1], label=f'W={W}, C=0.5')
    plt.plot(t, sol[:, 2], label=f'W={W}, C=0.1')

plt.xlabel('Час (с)')
plt.ylabel('Тиск P(t) (мм рт. ст.)')
plt.title('Зміна тиску P(t) для різних значень W та C')
plt.legend()
plt.grid(True)
plt.show()
