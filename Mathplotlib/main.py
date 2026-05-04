import numpy as np
from scipy import stats

print("-"*70)
print("1. Випадкові величини")
print("-"*70)

print("Дискретна випадкова величина")
dice = [1,2,3,4,5,6,7,8,9]
print(f"Діапазон {dice}")

print("Неперервна випадкова величина")
temp = np.random.uniform(1,20,1000)
print(f"Діапазон {temp.min():.2f}°C - {temp.max():.2f}°C")

print("-"*70)
print("2. Статистичні середні показники")
print("-"*70)
# Для прикладу використаємо дані про оцінки студентів
grades = np.array([85, 92, 78, 88, 95, 82, 89, 91, 76, 94, 87, 83])
print(f"\nДані: Оцінки студентів = {grades}")

# 2.1 СЕРЕДНЄ (МАТЕМАТИЧНЕ СПОДІВАННЯ)
mean = np.mean(grades)
print(f"\n2.1 СЕРЕДНЄ: {mean:.2f}")
print(f"    Формула: (x₁ + x₂ + ... + xₙ) / n")
print(f"    Обчислення: ({' + '.join(map(str, grades[:3]))} + ...) / {len(grades)} = {mean:.2f}")

# 2.2 МОДА (НАЙЧАСТІШЕ ЗНАЧЕННЯ)
print(f"\n2.2 МОДА")
grades_mode = np.array([85, 92, 78, 85, 95, 82, 85, 91, 76, 94, 87, 83])
print(f"    Дані: {grades_mode}")
mode_value = stats.mode(grades_mode, keepdims=True).mode[0]
mode_count = stats.mode(grades_mode, keepdims=True).count[0]
print(f"    Мода: {mode_value} (з'являється {mode_count} раз)")
print(f"    Інтерпретація: Найчастіше зустрічається оцінка {mode_value}")

# 2.3 МЕДІАНА (СЕРЕДНЄ ЗНАЧЕННЯ)
sorted_grades = np.sort(grades)
median = np.median(grades)
print(f"\n2.3 МЕДІАНА: {median:.2f}")
print(f"    Відсортовані дані: {sorted_grades}")
print(f"    Інтерпретація: 50% студентів мають оцінку ≤ {median:.2f}, 50% > {median:.2f}")

# 2.4 ДИСПЕРСІЯ (РОЗКИД ЗНАЧЕНЬ)
variance = np.var(grades, ddof=1)  # ddof=1 для вибіркової дисперсії
print(f"\n2.4 ДИСПЕРСІЯ: {variance:.2f}")
print(f"    Формула: Σ(xᵢ - x̄)² / (n-1)")
deviations = grades - mean
squared_deviations = deviations ** 2
print(f"    Приклад розрахунку:")
print(f"    1. Відхилення від середнього: {deviations[:3]} ...")
print(f"    2. Квадрати відхилень: {squared_deviations[:3]} ...")
print(f"    3. Дисперсія = {variance:.2f}")

# 2.5 СЕРЕДНЬОКВАДРАТИЧНЕ ВІДХИЛЕННЯ
std_dev = np.std(grades, ddof=1)
print(f"\n2.5 СЕРЕДНЬОКВАДРАТИЧНЕ ВІДХИЛЕННЯ: {std_dev:.2f}")
print(f"    Формула: σ = √Дисперсія")
print(f"    σ = √{variance:.2f} = {std_dev:.2f}")
print(f"    Інтерпретація: Середня величина розкиду даних від середнього")

# Правило 3-сигма для нормального розподілу
print(f"\n    Правило 3-сигма (68-95-99.7%):")
print(f"    - 68% даних: [{mean - std_dev:.2f}; {mean + std_dev:.2f}]")
print(f"    - 95% даних: [{mean - 2*std_dev:.2f}; {mean + 2*std_dev:.2f}]")
print(f"    - 99.7% даних: [{mean - 3*std_dev:.2f}; {mean + 3*std_dev:.2f}]")