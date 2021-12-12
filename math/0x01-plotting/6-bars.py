#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

"""fruit is a matrix representing the number of fruit """


np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

peoples = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

fig, axes = plt.subplots()
fig.suptitle("Number of Fruit per Person")

axes.set_ylabel("Quantity of Fruit")
axes.set_ylim(0, 80)
axes.set_yticks(range(0, 81, 10))

for row, qty in enumerate(fruit):
    axes.bar(peoples, qty, width=0.5,
            bottom=[sum(fruit[:row, col]) for col in range(len(qty))],
            label=fruits[row],
            color=colors[row])

axes.legend()
plt.show()
