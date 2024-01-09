import matplotlib.pyplot as plt
import numpy as np

# Define the parametric equations for the crescent moon shape
def parametric_crescent(t, a, b):
    x = np.cos(t)
    y = a * np.sin(t) - b * np.sin(2 * t)
    return x, y

# Define the values for the coefficients 'a' and 'b'
a_value = 1.5 # Adjust this value to modify the overall size of the crescent
b_value = -0.7  # Adjust this value to modify the depth of the crescent

# Generate values for the parametric curve (t from 0 to 2*pi)
t_values = np.linspace(0, 2 * np.pi, 1000)
x_values, y_values = parametric_crescent(t_values, a_value, b_value)

# Plotting the parametric curve representing the crescent moon shape
plt.figure(figsize=(6, 6))
plt.plot(x_values, y_values, 'b')
plt.title('Parametric Curve of a Crescent Moon Shape')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axis('equal')
plt.show()
