import matplotlib.pyplot as plt
import numpy as np


class_1 = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 2.5), (4.1, 1), (3, 2), (2, 2.5), (3, 2.5), (4, 3)]
class_2 = [(5.9, 1), (6, 2), (6, 2.5), (7, 1), (7, 2), (7, 2.5), (8, 1), (8, 2), (8, 2.5), (9, 1)]
class_3 = [(1, 8.5), (1, 9), (1, 10), (2, 8.5), (2, 9), (2, 10), (3, 8.5), (3, 10), (3, 9), (4, 8)]

plt.scatter(*zip(*class_1), c='red', label='Class 1')
plt.scatter(*zip(*class_2), c='blue', label='Class 2')
plt.scatter(*zip(*class_3), c='green', label='Class 3')

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.grid()
plt.show()


