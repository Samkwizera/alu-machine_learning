#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
bins = np.arange(0, 101, 10)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.hist(student_grades, bins=bins, color='blue', edgecolor='black')
plt.xlim(1, 100)
plt.xticks(bins)
plt.show()
