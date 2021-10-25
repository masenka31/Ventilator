# If starting from Git Bash:
# 1) don't forget to activate the virtual environment: source Ventilator.py/Scripts/activate
# 2) Start python with winpty python
import matplotlib.pyplot as plt
import numpy as np

# test matplotlib and numpy packages
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# import train.csv data
import pandas as pd
data = pd.read_csv('Ventilator.jl\\data\\train.csv')
data.head()

