import matplotlib.pyplot as plt
import numpy as np

print("="*10, " N1 " , "="*10)
months = ["Січ", "Лют", "Бер", "Кві", "Тра", "Чер", "Лип", "Сер", "Вер", "Жов", "Лис", "Гру"]
plan = [100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]
fact = [90, 110, 125, 135, 155, 150, 165, 175, 185, 195, 205, 215]

plt.figure()
plt.plot(months, plan, label="План")
plt.plot(months, fact, label="Факт")
plt.title("План vs Факт продажів")
plt.xlabel("Місяці")
plt.ylabel("Продажі")
plt.legend()
plt.show()


print("="*10, " N2 " , "="*10)
ages = np.random.randint(18, 70, 100)

plt.figure()
plt.hist(ages, bins=10)
plt.axvline(np.mean(ages), color="red")
plt.title("Розподіл віку")
plt.xlabel("Вік")
plt.ylabel("Кількість")
plt.show()


print("="*10, " N3 " , "="*10)
group1 = np.random.normal(70, 10, 30)
group2 = np.random.normal(75, 15, 30)
group3 = np.random.normal(80, 20, 30)

plt.figure()
plt.boxplot([group1, group2, group3], tick_labels=["Група 1", "Група 2", "Група 3"])
plt.title("Розподіл оцінок")
plt.xlabel("Групи")
plt.ylabel("Оцінки")
plt.show()


print("="*10, " N4 " , "="*10)
days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
temp = [20, 22, 21, 23, 24, 25, 23]
humidity = [60, 65, 63, 70, 72, 75, 73]

plt.figure()
plt.plot(days, temp, label="Температура")
plt.plot(days, humidity, label="Вологість")
plt.title("Погода за тиждень")
plt.xlabel("Дні")
plt.ylabel("Значення")
plt.xticks(rotation=45)
plt.legend()
plt.show()


print("="*10, " N5 " , "="*10)
hours = list(range(24))
load = np.random.randint(20, 100, 24)

plt.figure()
plt.plot(hours, load)
plt.fill_between(hours, load, alpha=0.3)
plt.title("Навантаження сервера")
plt.xlabel("Години")
plt.ylabel("Завантаження")
plt.grid()
plt.show()


print("="*10, " N6 " , "="*10)
x = np.arange(12)

conversion = np.random.rand(12) * 100
retention = np.random.rand(12) * 100
avg_check = np.random.rand(12) * 1000
orders = np.random.rand(12) * 500

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(x, conversion)
axs[0, 0].set_title("Конверсія")

axs[0, 1].plot(x, retention)
axs[0, 1].set_title("Утримання")

axs[1, 0].plot(x, avg_check)
axs[1, 0].set_title("Середній чек")

axs[1, 1].plot(x, orders)
axs[1, 1].set_title("Кількість замовлень")

plt.suptitle("Метрики продукту")
plt.tight_layout()
plt.show()