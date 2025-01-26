import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Набір параметрів
params_list = [
    {'alpha': 0.01, 'beta': 0.5, 'epsilon': 0.5, 'delta': 0.01},
    {'alpha': 0.02, 'beta': 0.4, 'epsilon': 0.6, 'delta': 0.02},
    {'alpha': 0.03, 'beta': 0.3, 'epsilon': 0.7, 'delta': 0.03}
]

# Початкові умови
initial_conditions = [
    {'x0': 40, 'y0': 9},
    {'x0': 50, 'y0': 5},
    {'x0': 30, 'y0': 15}
]

def lotka_volterra(z, t, alpha, beta, epsilon, delta):
    x, y = z
    dxdt = epsilon * x - alpha * x * y
    dydt = delta * x * y - beta * y
    return [dxdt, dydt]

t_start = 0
t_end = 200
t_eval = np.linspace(t_start, t_end, 2000)  # Крок інтегрування

for idx, params in enumerate(params_list):
    alpha = params['alpha']
    beta = params['beta']
    epsilon = params['epsilon']
    delta = params['delta']
    
    for ic_idx, ic in enumerate(initial_conditions):
        x0 = ic['x0']
        y0 = ic['y0']
        
        # Розв'язуємо систему за допомогою odeint
        sol = odeint(
            lotka_volterra,
            [x0, y0],
            t_eval,
            args=(alpha, beta, epsilon, delta)
        )
        
        x = sol[:, 0]
        y = sol[:, 1]
        t = t_eval
        
        # Побудова графіків x(t) та y(t)
        plt.figure(figsize=(14, 5))
        
        # Графік x(t) та y(t)
        plt.subplot(1, 2, 1)
        plt.plot(t, x, label='Жертви (x)')
        plt.plot(t, y, label='Хижаки (y)')
        plt.title(f'Часові залежності (Набір параметрів {idx+1}, Початкові умови {ic_idx+1})')
        plt.xlabel('Час t')
        plt.ylabel('Чисельність популяції')
        plt.legend()
        
        # Фазовий портрет
        plt.subplot(1, 2, 2)
        plt.plot(x, y)
        plt.title(f'Фазовий портрет (Набір параметрів {idx+1}, Початкові умови {ic_idx+1})')
        plt.xlabel('Жертви (x)')
        plt.ylabel('Хижаки (y)')
        
        plt.tight_layout()
        plt.show()


for idx, params in enumerate(params_list):
    alpha = params['alpha']
    beta = params['beta']
    epsilon = params['epsilon']
    delta = params['delta']
    
    for ic_idx, ic in enumerate(initial_conditions):
        x0 = ic['x0']
        y0 = ic['y0']
        
        # Розв'язуємо систему за допомогою odeint
        sol = odeint(
            lotka_volterra,
            [x0, y0],
            t_eval,
            args=(alpha, beta, epsilon, delta)
        )
        
        x = sol[:, 0]
        y = sol[:, 1]
        t = t_eval
        
        # Знаходимо піки
        peaks_x, _ = find_peaks(x)
        peaks_y, _ = find_peaks(y)
        
        # Обчислюємо періоди
        period_x = np.diff(t[peaks_x]).mean() if len(peaks_x) > 1 else np.nan
        period_y = np.diff(t[peaks_y]).mean() if len(peaks_y) > 1 else np.nan
        
        print(f'Набір параметрів {idx+1}, Початкові умови {ic_idx+1}:')
        print(f'Період коливань жертв (x): {period_x:.2f}')
        print(f'Період коливань хижаків (y): {period_y:.2f}\n')


for idx, params in enumerate(params_list):
    alpha = params['alpha']
    beta = params['beta']
    epsilon = params['epsilon']
    delta = params['delta']
    
    # Стаціонарні точки
    x_sm = beta / delta
    y_sm = epsilon / alpha
    print(f'Набір параметрів {idx+1}: Стаціонарні точки x̄ = {x_sm:.2f}, ȳ = {y_sm:.2f}')
    
    # Варіюємо початкові умови навколо стаціонарних точок
    deviations = [-0.1, 0, 0.1]
    for dx in deviations:
        for dy in deviations:
            x0 = x_sm * (1 + dx)
            y0 = y_sm * (1 + dy)
            
            # Розв'язуємо систему за допомогою odeint
            sol = odeint(
                lotka_volterra,
                [x0, y0],
                t_eval,
                args=(alpha, beta, epsilon, delta)
            )
            
            x = sol[:, 0]
            y = sol[:, 1]
            t = t_eval
            
            # Фазовий портрет
            plt.plot(x, y, label=f'Δx={dx}, Δy={dy}')
    
    plt.title(f'Фазові траєкторії навколо стаціонарної точки (Набір параметрів {idx+1})')
    plt.xlabel('Жертви (x)')
    plt.ylabel('Хижаки (y)')
    plt.legend()
    plt.show()

# Параметри зменшених взаємодій
alpha_small = 1e-6  # дуже мале значення
delta_small = 1e-6  # дуже мале значення
beta = 0.5
epsilon = 0.5

# Початкові умови
x0 = 40
y0 = 9

# Часовий інтервал
t_start = 0
t_end = 200
t_eval = np.linspace(t_start, t_end, 2000)

# Функція для системи Лотки-Вольтерри з малими взаємодіями
def lotka_volterra_small(z, t):
    x, y = z
    dxdt = epsilon * x - alpha_small * x * y
    dydt = delta_small * x * y - beta * y
    return [dxdt, dydt]

# Розв'язок системи з малими взаємодіями
sol_small = odeint(
    lotka_volterra_small,
    [x0, y0],
    t_eval
)

x_small = sol_small[:, 0]
y_small = sol_small[:, 1]
t = t_eval

# Функція для моделі природного росту
def natural_growth(z, t):
    x, y = z
    dxdt = epsilon * x
    dydt = -beta * y
    return [dxdt, dydt]

# Розв'язок системи природного росту
sol_natural = odeint(
    natural_growth,
    [x0, y0],
    t_eval
)

x_natural = sol_natural[:, 0]
y_natural = sol_natural[:, 1]

# Побудова графіків
plt.figure(figsize=(14, 5))

# Графік чисельності жертв
plt.subplot(1, 2, 1)
plt.plot(t, x_small, label='Жертви з взаємодією')
plt.plot(t, x_natural, '--', label='Жертви без взаємодії')
plt.title('Чисельність жертв')
plt.xlabel('Час t')
plt.ylabel('Чисельність жертв')
plt.legend()

# Графік чисельності хижаків
plt.subplot(1, 2, 2)
plt.plot(t, y_small, label='Хижаки з взаємодією')
plt.plot(t, y_natural, '--', label='Хижаки без взаємодії')
plt.title('Чисельність хижаків')
plt.xlabel('Час t')
plt.ylabel('Чисельність хижаків')
plt.legend()

plt.tight_layout()
plt.show()
