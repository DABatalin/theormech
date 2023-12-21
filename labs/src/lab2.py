import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

points_count = 100

R = 4
r = 1

phi = 2*t + 10
theta = 3*t - 0.6


# def DrawTorus(R, r, x_offset=0, y_offset=0):
#     torus_theta = np.linspace(0, 2*np.pi, points_count)
#     torus_phi = np.linspace(0, 2*np.pi, points_count)
#     torus_theta, torus_phi = np.meshgrid(torus_theta, torus_phi)
#     torus_X = r * np.sin(torus_phi) + x_offset
#     torus_Y = (R + r * np.cos(torus_phi)) * np.cos(torus_theta) + y_offset
#     torus_Z = (R + r * np.cos(torus_phi)) * np.sin(torus_theta)
#     return [torus_X, torus_Y, torus_Z]


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

def draw_axes(ax, length=5):
    ax.quiver(0, 0, 0, length, 0, 0, color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, length, 0, color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, length, color='black', arrow_length_ratio=0.1)

ring_coords1 = DrawRing(R)
ring_coords2 = DrawRing(R+2*r)
point_coords = DrawRing(r, y_offset=R+r)
# torus_coords = DrawTorus(R + 1, r)

# print(np.shape(ring_coords1))
# print(np.shape(torus_coords))
# print(torus_coords[1][1])

fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d([-7.0, 7.0])
ax.set_ylim3d([-7.0, 7.0])
ax.set_zlim3d([-5.0, 5.0])
draw_axes(ax, length=10)


ring1, = ax.plot(*ring_coords1, c='blue')
ring2, = ax.plot(*ring_coords2, c='blue')
point, = ax.plot(*point_coords, c="red")
# torus = [ax.plot_surface(*torus_coords, color='b', alpha=0.5)]


def update(i):
    ring1.set_data(RotateXY(ring_coords1, i)[:2])
    ring1.set_3d_properties(ring_coords1[2])

    # torus[0].remove()
    # torus[0] = ax.plot_surface(*RotateXY([torus_x, torus_y], i), torus_z, color='b', alpha=0.5)
    

    ring2.set_data(RotateXY(ring_coords2, i)[:2])
    ring2.set_3d_properties(ring_coords2[2])
	
    c = RotateXY(RotateYZ(point_coords, i), i)
    point.set_data(c[:2])
    point.set_3d_properties(c[2])

ani = FuncAnimation(fig, update, frames=len(t), interval=40, repeat=False)

plt.show()