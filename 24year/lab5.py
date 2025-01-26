import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pharmacokinetic_model(C, t, k1, k23, k32, k4):
    C1, C2, C3 = C
    dC1_dt = -k1 * C1
    dC2_dt = k1 * C1 - (k23 + k4) * C2 + k32 * C3
    dC3_dt = k23 * C2 - k32 * C3
    return [dC1_dt, dC2_dt, dC3_dt]

# Початкові умови
C1_0 = 100  # Доза препарату в місці введення
C2_0 = 0    # Початкова концентрація в крові
C3_0 = 0    # Початкова концентрація в органі-мішені
initial_conditions = [C1_0, C2_0, C3_0]

# Часовий інтервал моделювання
t = np.linspace(0, 50, 500)  # Від 0 до 50 одиниць часу, 500 точок

# Набори констант
parameters = [
    {'k1': 0.5, 'k23': 0.3, 'k32': 0.1, 'k4': 0.05, 'label': 'Базові значення'},
    {'k1': 1.0, 'k23': 0.3, 'k32': 0.1, 'k4': 0.1,  'label': 'Швидке всмоктування та виведення'},
    {'k1': 0.2, 'k23': 0.5, 'k32': 0.05, 'k4': 0.02,'label': 'Повільне всмоктування та накопичення'}
]

# Словник для збереження результатів
results = {}

for param in parameters:
    # Розпаковуємо константи
    k1 = param['k1']
    k23 = param['k23']
    k32 = param['k32']
    k4 = param['k4']
    label = param['label']
    
    # Розв'язуємо систему
    solution = odeint(pharmacokinetic_model, initial_conditions, t, args=(k1, k23, k32, k4))
    C1 = solution[:, 0]
    C2 = solution[:, 1]
    C3 = solution[:, 2]
    
    # Зберігаємо результати
    results[label] = {'C1': C1, 'C2': C2, 'C3': C3, 'k1': k1, 'k23': k23, 'k32': k32, 'k4': k4}
    
    # Побудова графіків
    plt.figure(figsize=(12, 6))
    plt.plot(t, C1, label='C1 (Місце введення)')
    plt.plot(t, C2, label='C2 (Кров)')
    plt.plot(t, C3, label='C3 (Орган-мішень)')
    plt.xlabel('Час')
    plt.ylabel('Концентрація')
    plt.title(f"Зміна концентрації препарату у часі: {label}\n"
              f"k1={k1}, k23={k23}, k32={k32}, k4={k4}")
    plt.legend()
    plt.grid()
    plt.show()

# Обчислення часу T0.5 та інших параметрів
for label, data in results.items():
    C1_0 = initial_conditions[0]
    C3 = data['C3']
    C3_max = np.max(C3)
    half_max = C3_max / 2
    indices = np.where(C3 >= half_max)[0]
    if len(indices) > 0:
        T_half = t[indices[-1]]  # Останній момент, коли C3 >= половини максимуму
        print(f"{label}:")
        print(f"  Максимальна концентрація в органі-мішені: {C3_max:.2f}")
        print(f"  Час T0.5 (коли концентрація зменшується до половини максимуму): {T_half:.2f}")
    else:
        print(f"{label}: Концентрація не досягає половини максимального значення.")
    
    fraction_reached = (C3_max / C1_0) * 100  # У відсотках
    index_C3_max = np.argmax(C3)
    time_C3_max = t[index_C3_max]
    print(f"  Частка препарату, що досягає органу-мішені: {fraction_reached:.2f}%")
    print(f"  Час досягнення максимальної концентрації в органі-мішені: {time_C3_max:.2f}\n")
