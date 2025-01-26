import math
import matplotlib.pyplot as plt

############################################################
# 1. читання файлу
############################################################
def read_xyz_file(file_path):
    atoms_data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                atom_name = parts[0]
                x, y, z = map(float, parts[1:])
                atoms_data.append((atom_name, x, y, z))
    return atoms_data #[('H', 0.0, 0.0, 0.0), ('O', 1.0, 0.0, 0.0), ('H', 0.0, 1.0, 0.0)]

############################################################
# 2. побудова матриці відстаней
############################################################
def build_distance_matrix(atoms_data):
    """
    повертає матрицю відстаней dist_matrix[i][j] = відстань між атомом i та j.
    """
    n = len(atoms_data)
    dist_matrix = [[0.0]*n for _ in range(n)]
    
    for i in range(n):
        x1, y1, z1 = atoms_data[i][1], atoms_data[i][2], atoms_data[i][3]
        for j in range(i+1, n):
            x2, y2, z2 = atoms_data[j][1], atoms_data[j][2], atoms_data[j][3]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) #евклідова відстань 
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix 
"""
результат:
sqrt(1+1+0) = sqrt(2) = 1.414

приклад
[0.0, 1.0, 1.0]
[1.0, 0.0, 1.414]
[1.0, 1.414, 0.0
"""

############################################################
# 3. визначення зв’язків
############################################################
def filter_covalent_bonds(dist_matrix, atoms_data, covalent_radii, tolerance=0.1):
    """
    повертає матрицю зв’язків (True/False) між атомами.
    критерій: distance <= (r_i + r_j + tolerance).
    """
    n = len(dist_matrix)
    bond_matrix = [[False]*n for _ in range(n)] #матриця з False
    
    for i in range(n):
        elem_i = atoms_data[i][0].upper()
        for j in range(i+1, n):
            elem_j = atoms_data[j][0].upper()
            distance = dist_matrix[i][j]
            
            # радіуси (якщо немає в словнику - 0.0)
            r_i = covalent_radii.get(elem_i, 0.0)
            r_j = covalent_radii.get(elem_j, 0.0)
            
            if 0.0 < distance <= (r_i + r_j + tolerance):
                bond_matrix[i][j] = True
                bond_matrix[j][i] = True
            """
            якщо відстань між атомами більше 0 та менше 
            або дорівнює сумі їхніх ковалентних радіусів 
            то між атомами утворюється зв'язок
            """
    
            """
            вхідні дані: 
            atoms_data = [
            ('H', 0.0, 0.0, 0.0),    # Атом 0
            ('H', 0.0, 0.0, 0.74),   # Атом 1
            ('O', 1.0, 0.0, 0.0)     # Атом 2

            dist_matrix = [
                [0.0, 0.74, 1.0],
                [0.74, 0.0, 1.3],
                [1.0, 1.3, 0.0]]

            covalent_radii = 
            {'H': 0.31, 'O': 0.66}

           обчислення:
            для атомів 0 (H) і 1 (H):
                відстань = 0.74, радіуси = 0.31 + 0.31 + 0.1
                оскільки 0.74 > 0.72, зв'язок не утворюється
            для атомів 0 (H) і 2 (O):
                відстань = 1.0, радіуси = 0.31 + 0.66 + 0.1
                оскільки 1.0 <= 1.07, зв'язок утворюється
            для атомів 1 (H) і 2 (O):
                оскільки 1.3 > 1.07, зв'язок не утворюється.

            вихідні:
            bond_matrix = [
            [False, False, True],  # зв'язок між 0 і 2
            [False, False, False], # жодних зв'язків для 1
            [True, False, False]   # зв'язок між 2 і 0
            ]
            """
    return bond_matrix

############################################################
# 4. пошук циклів (5- або 6-членних) у матриці зв’язків
############################################################
def find_rings_of_size(bond_matrix, atoms_data, ring_size=5, element_criteria=None):
    if element_criteria is None:
        element_criteria = {}
    
    n = len(atoms_data)
    visited_cycles = set()
    found_rings = []

    def dfs(path, start):
        current = path[-1]
        if len(path) > ring_size:
            return
        
        for nxt in range(n):
            if bond_matrix[current][nxt]:
                if nxt == start and len(path) == ring_size:
                    """
                    ми починаємо з кожного атома, перевіряючи його сусідів, доки:
                    не досягнемо кільця потрібного розміру
                    не повернемося до початкового атома
                    """
                    cycle_sorted = tuple(sorted(path))
                    if cycle_sorted not in visited_cycles:
                        visited_cycles.add(cycle_sorted)
                        if check_elements(path, atoms_data, element_criteria):
                            found_rings.append(path[:])
                elif (nxt not in path) and (len(path) < ring_size):
                    path.append(nxt)
                    dfs(path, start)
                    path.pop()
    
    def check_elements(path, atoms_data, criteria):
        """ перевіряє, чи атоми в path відповідають критеріїю element_criteria """
        from collections import Counter
        cnt = Counter()
        for idx in path:
            elem = atoms_data[idx][0].upper()[0]
            cnt[elem] += 1

        for elem, required_count in criteria.items():
            if cnt[elem] != required_count:
                return False

        if sum(cnt.values()) == sum(criteria.values()):
            return True
        return False

    for start_atom in range(n):
        dfs([start_atom], start_atom)
    
    return found_rings

############################################################
# 5. «впізнавання» фрагментів та присвоєння імен
############################################################
def analyze_bond_matrix(bond_matrix, atoms_data):
    """
    алгоритм:
    1) шукаємо фосфор (якщо є).
    2) шукаємо 5-членне кільце.
    3) шукаємо 6-членне (або 5-членне) кільце з N (основа).
    4) присвоюємо імена (C1', C2', ..., N1, N9, тощо) - дуже спрощено!
    
    повертає список словників із ключами:
        index, name, nucleotide, residue_number, x, y, z
    """
    n = len(atoms_data)
    recognized = [None]*n
    element_list = [a[0].upper() for a in atoms_data]
    
    p_indices = [i for i, el in enumerate(element_list) if el.startswith("P")]
    for i in p_indices:
        recognized[i] = "P"

        neighbors = [j for j in range(n) if bond_matrix[i][j]]
        o_neighbors = [j for j in neighbors if element_list[j].startswith("O")]
        if len(o_neighbors) >= 2:
            recognized[o_neighbors[0]] = "OP1"
        if len(o_neighbors) >= 3:
            recognized[o_neighbors[1]] = "OP2"
        if len(o_neighbors) >= 4:
            recognized[o_neighbors[2]] = "O5'"
            recognized[o_neighbors[3]] = "O3'"

    sugar_rings = find_rings_of_size(
        bond_matrix,
        atoms_data,
        ring_size=5,
        element_criteria={"C":4, "O":1}
    )

    if sugar_rings:
        sugar = sugar_rings[0]
        # нумеруємо як C1', C2', C3', C4', O4' (спрощено)
        # реально треба визначати порядок, але тут припустимо що за індексами як є
        # або вибрати, де O і позначити його O4'
        # упорядкуємо sugar по елементу, щоб O ішов останнім
        # (або знайдемо точний індекс O)
        o_idx = None
        c_indices = []
        for idx in sugar:
            if element_list[idx].startswith("O"):
                o_idx = idx
            else:
                c_indices.append(idx)

        if len(c_indices) == 4:
            recognized[c_indices[0]] = "C1'"
            if len(c_indices) > 1:
                recognized[c_indices[1]] = "C2'"
            if len(c_indices) > 2:
                recognized[c_indices[2]] = "C3'"
            if len(c_indices) > 3:
                recognized[c_indices[3]] = "C4'"
        if o_idx is not None:
            recognized[o_idx] = "O4'"
        
        if len(c_indices) == 4:
            c4_prime = c_indices[3]
            c4_neighbors = [j for j in range(n) if bond_matrix[c4_prime][j]]
            for candi in c4_neighbors:
                if element_list[candi].startswith("C") and (candi not in sugar):
                    recognized[candi] = "C5'"

    base_rings_6 = find_rings_of_size(
        bond_matrix,
        atoms_data,
        ring_size=6,
        element_criteria={"C":4, "N":2}
    )

    if base_rings_6:
        ring = base_rings_6[0]
        n_list = []
        c_list = []
        for idx in ring:
            if element_list[idx].startswith("N"):
                n_list.append(idx)
            else:
                c_list.append(idx)

        if len(ring) == 6:
            ring_sorted = sorted(ring)
            recognized[ring_sorted[0]] = "N1"
            if len(ring_sorted) > 1:
                recognized[ring_sorted[1]] = "C2"
            if len(ring_sorted) > 2:
                recognized[ring_sorted[2]] = "N3"
            if len(ring_sorted) > 3:
                recognized[ring_sorted[3]] = "C4"
            if len(ring_sorted) > 4:
                recognized[ring_sorted[4]] = "C5"
            if len(ring_sorted) > 5:
                recognized[ring_sorted[5]] = "C6"

    recognized_atoms = []
    residue_name = "DUNK"
    if base_rings_6:
        residue_name = "DC"

    for i, (atom_name, x, y, z) in enumerate(atoms_data):
        assigned = recognized[i]
        if assigned is None:
            assigned = atom_name
        
        recognized_atoms.append({
            "index": i+1,
            "name": assigned,
            "nucleotide": residue_name,
            "residue_number": 1,
            "x": x,
            "y": y,
            "z": z
        })
    
    return recognized_atoms

############################################################
# 6. візуалізація
############################################################
ATOM_PROPERTIES = {
    "C": {"color": "black",    "size": 120},
    "O": {"color": "red",      "size": 130},
    "N": {"color": "blue",     "size": 130},
    "H": {"color": "lightgray","size": 80},
    "P": {"color": "orange",   "size": 150}
}
DEFAULT_PROPS = {"color": "gray", "size": 100}

def visualize_atoms_with_bonds(atoms_data, bond_matrix, recognized_atoms):
    """
    Виводить 3D-графіку:
     - атоми різного кольору, з додатковою прозорістю,
     - підписані присвоєними іменами,
     - зі зв’язками (лініями) між атомами, де bond_matrix[i][j] = True.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    used_labels = set()
    n = len(atoms_data)
    
    for i, (atom_name, x, y, z) in enumerate(atoms_data):
        element_symbol = atom_name[0].upper()
        props = ATOM_PROPERTIES.get(element_symbol, DEFAULT_PROPS)
        color = props["color"]
        size = props["size"]
        alpha = 0.2  # додаємо прозорість
        
        if element_symbol not in used_labels:
            label = element_symbol
            used_labels.add(element_symbol)
        else:
            label = None
        
        # відобразимо точку
        ax.scatter(x, y, z, c=color, s=size, label=label, alpha=alpha)
        
        # підпис (номенклатурне ім'я)
        assigned_name = recognized_atoms[i]["name"]
        ax.text(x, y, z, assigned_name, fontsize=7)
    
    # малюємо зв’язки
    for i in range(n):
        x1, y1, z1 = atoms_data[i][1], atoms_data[i][2], atoms_data[i][3]
        for j in range(i+1, n):
            if bond_matrix[i][j]:
                x2, y2, z2 = atoms_data[j][1], atoms_data[j][2], atoms_data[j][3]
                ax.plot([x1, x2], [y1, y2], [z1, z2], color="gray", linewidth=1)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Nucleotide Fragment Visualization")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


############################################################
# 7. повний pipeline у main
############################################################
if __name__ == "__main__":

    xyz_file = "main/5_r9_xyz_dftV3#r9_085_DaabB.xyz"
    atoms_data = read_xyz_file(xyz_file)
    dist_matrix = build_distance_matrix(atoms_data)
    
    covalent_radii = {
        "H": 0.31,
        "C": 0.77,
        "N": 0.74,
        "O": 0.66,
        "P": 1.07
    }
    bond_matrix = filter_covalent_bonds(dist_matrix, atoms_data, covalent_radii, tolerance=0.1)
    recognized_atoms = analyze_bond_matrix(bond_matrix, atoms_data) 
    visualize_atoms_with_bonds(atoms_data, bond_matrix, recognized_atoms)
    with open("output.pdb", "w") as pdb:
        for atom in recognized_atoms:
            pdb.write(
                f"ATOM  {atom['index']:>5} {atom['name']:<4}"
                f" {atom['nucleotide']:<3} {atom['residue_number']:>4}    "
                f"{atom['x']:>8.3f}{atom['y']:>8.3f}{atom['z']:>8.3f}"
                f"  1.00  0.00\n"
            )
    print("готово! перевірте output.pdb та графік.")