import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Создаем данные для тора
torus_theta = np.linspace(0, 2*np.pi, 100)
torus_phi = np.linspace(0, 2*np.pi, 100)
torus_theta, torus_phi = np.meshgrid(torus_theta, torus_phi)
torus_R = 5  # Радиус тора
torus_r = 1  # Радиус сечения тора

torus_x = (torus_R + torus_r * np.cos(torus_phi)) * np.cos(torus_theta)
torus_z = (torus_R + torus_r * np.cos(torus_phi)) * np.sin(torus_theta)
torus_y = torus_r * np.sin(torus_phi)

# Создаем 3D-график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-7.0, 7.0])
ax.set_ylim3d([-7.0, 7.0])
ax.set_zlim3d([-5.0, 5.0])

# Рисуем тор с прозрачностью
ax.plot_surface(torus_x, torus_y, torus_z, color='b', alpha=0.5)

# Настройка отображения
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Тор в Matplotlib с прозрачностью')

# Показываем график
plt.show()
