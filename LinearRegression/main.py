#Для Лінійної регресії
from sklearn.linear_model import LinearRegression
# Для роботи з масивами
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Для метрики якості
from sklearn.model_selection import train_test_split

print ("Робота з Регресією")
area = np.array([50,100,150,200,250]).reshape(-1,1) # з рядка робить стовпець
price = np.array ([200,250,300,350,400])
print (area)
print (price)

#Створили модель
model = LinearRegression()

#Передали дані
model.fit(area,price)

#формула для навчання y = b0 + b1 * x

#нова площа
new_area = np.array([[180]])

#передбачення яка буде ціна для нової площі
prediction = model.predict(new_area)
print (f"площа {new_area[0,0]}м2 -> передбачена ціна {prediction[0]:0f} тис. грн")
print(f"формула: ціна = {model.intercept_:.2f} + {model.coef_[0]:.2f} * площа")

print("="*20)
#будуємо предикат щоб отримати передбачення
predictions = model.predict(area)

#середня абсолютна помилка
#шукаємо різницю між ціною і прогнозом
#без знаку по модулю))
#наскільки модель буде помилятися
mea = mean_absolute_error(price, predictions)

#середньо квадратична помилка
#помилки підносяться до квадрату
#якщо помилки більші то вони будуть важливішими
rmse = np.sqrt(mea)

#коеф детермінації
#наскільки добре модель пояснює дані
r2 = r2_score(price, predictions)

print (f"MEA: {mea:.2f} (середня помилка)")
print (f"RMSE: {rmse:.2f} (коренева квадратична помилка)")
print (f"R^2: {r2:.2f} (якість моделі)")

print("="*20)
#список даних для моделі
X = np.array([
    [50,2,30],#50 - площа, 2 - кімнати, 30 - років
    [100,3,20],
    [150,4,10],
    [200,5,5],
    [250,6,0],
])

#ціни
Y = np.array([100,150,200,300,400])

#
model_multy = LinearRegression()
model_multy.fit(X, Y)

#передбачення на нове житло
new_house = np.array([[120,3,15]])
#скільки буде коштувати житло 120 кв, 3 кімнати, 15 років
price_pred = model_multy.predict(new_house)

#
print(f"Новий будинок 120 кв, 3 кімнати, 15 років - ціна {price_pred[0]:.0f} тис грн")
print(f"Вплив кожної ознаки на ціну:")
for i, name in enumerate(["Площа","Кімнати","Вік"]):
    print(f"\t{name}: {model_multy.coef_[i]:.2f}")



