import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

points_count = 100

R = 6
r = 0.5

phi = 2*t + 10
theta = 3*t - 0.6

def DrawRing(radius, x_offset=0, y_offset=0):
    ring_X = np.zeros((points_count))+x_offset
    ring_Y = radius*np.cos(np.linspace(0, 2*np.pi, points_count))+y_offset
    ring_Z = radius*np.sin(np.linspace(0, 2*np.pi, points_count))
    return [ring_X, ring_Y, ring_Z]

def RotateXY(obj, i):
    tmp = copy.deepcopy(obj)
    rotation_matrix = [[np.cos(phi[i]), -np.sin(phi[i])],
                       [np.sin(phi[i]), np.cos(phi[i])]]
    tmp[:2] = np.dot(rotation_matrix, obj[:2])
    return tmp

def RotateYZ(obj, i):
    tmp = copy.deepcopy(obj)
    rotation_matrix = [[np.cos(theta[i]), -np.sin(theta[i])],
                       [np.sin(theta[i]), np.cos(theta[i])]]
    tmp[1:3] = np.dot(rotation_matrix, obj[1:3])
    return tmp

ring_coords1 = DrawRing(R)
ring_coords2 = DrawRing(R+2*r)
point_coords = DrawRing(r, y_offset=R+r)

fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d([-7.0, 7.0])
ax.set_ylim3d([-7.0, 7.0])
ax.set_zlim3d([-5.0, 5.0])

ring1, = ax.plot(*ring_coords1, c='blue')
ring2, = ax.plot(*ring_coords2, c='blue')
point, = ax.plot(*point_coords, c="red")

def update(i):
    ring1.set_data(RotateXY(ring_coords1, i)[:2])
    ring1.set_3d_properties(ring_coords1[2])

    ring2.set_data(RotateXY(ring_coords2, i)[:2])
    ring2.set_3d_properties(ring_coords2[2])
	
    c = RotateXY(RotateYZ(point_coords, i), i)
    point.set_data(c[:2])
    point.set_3d_properties(c[2])

ani = FuncAnimation(fig, update, frames=len(t), interval=40, repeat=False)

plt.show()