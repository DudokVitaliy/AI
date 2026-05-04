import numpy as np

#myarray = np.array([1, 2, 3, 4, 5])
#print(myarray)

#myarray2 = np.array([[1, 2, 3], [4, 5, 6]])
#print(myarray2)

#zeros = np.zeros((3,3))
#print(zeros)

#ones = np.ones((3,3))
#print(ones)

#arr = np.arange(0, 10, 2)
#print(arr)

#arr = np.linspace(0, 1, 5)
#print(arr)

#myarr = np.array([[2,3,6],[4,6,6]])
#print(myarr)
#print(myarr.shape) #розмір
#print(myarr.ndim) #к-сть вимірів
#print(myarr.dtype) #тип данних
#print(myarr.size) #к-сть елем

#mylist = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#print(mylist[0])
#print(mylist[1:4])
#print(mylist[:4])
#print(mylist[4:])
#print(mylist[::2]) #кожен 2-гий

#my2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#print(my2d)
#print(my2d[1:,:2])

#a = np.array([1,2,3,4,5])
#b = np.array([6,7,8,9,0])
#print("a:", a)
#print("b:", b)
#print("a + b:", a + b)
#print("a - b:", a - b)
#print("a * b:", a * b)
#print("a / b:", a / b)
#print("a // b:", a // b)
#print("a % b:", a % b)
#print("a ** b:", a ** b)
#print("a == b:", a == b)
#print("a != b:", a != b)

#Статистичні методи
data = np.array([2,3,4,4,5,6,7])
print(data)
print(data.mean())
print(np.median(data))
print(np.max(data))
print(np.min(data))