import numpy as np
import copy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

m = 1
Jz = 0.5
R = 0.4
c = 2
g = 9.81
M = 5
k = 2

phi0 = 0
theta0 = math.pi / 2
d_phi0 = 0
d_theta0 = 0
y0 = [phi0, theta0, d_phi0, d_theta0]

Steps = 501
t_fin = 10
t = np.linspace(0, t_fin, Steps)

def odesys(y, t, m, Jz, R, c, g):
    dy = np.zeros(4)
    dy[0] = y[2] 
    dy[1] = y[3]

    dy[2] = -(m*R*R*y[2]*y[3]*np.sin(2*y[1])+c*y[0])/(Jz+m*R*R*np.sin(y[1])*np.sin(y[1]))
    dy[3] = (R*y[2]*y[2]*np.sin(y[1])*np.cos(y[1])-g*np.sin(y[1]))/R
    return dy

Y = odeint(odesys, y0, t, (m, Jz, R, c, g))

phi = Y[:,0]
theta = Y[:,1]
dphi = Y[:,2]
dtheta = Y[:,3]
ddphi = [odesys(y, t, m, Jz, R, c, g)[2] for y,t in zip(Y,t)]
ddtheta = [odesys(y, t, m, Jz, R, c, g)[3] for y,t in zip(Y,t)]

ROZ = (M + m)*g + m*R*(ddtheta*np.sin(theta) + dtheta**2 * np.cos(theta))
RBX = -m*R*(ddphi*np.sin(theta) + 2*dphi*dtheta*np.cos(theta)) * (1 - k*np.cos(theta)) / 2
RBY = -(m/2)*k*g*np.sin(theta) + (m*R/2)*(ddtheta*np.cos(theta) - (dtheta**2 + dphi**2) * np.sin(theta) - k*(ddtheta + dphi**2*np.sin(theta)*np.cos(theta)))
RB = np.sqrt(RBX**2 + RBY**2)                               

points_count = 100

R_draw = 8
r_draw = 0.5

def DrawRing(radius, x_offset=0, y_offset=0, z_offset = 0):
    ring_X = np.zeros((points_count))+x_offset
    ring_Y = radius*np.cos(np.linspace(0, 2*np.pi, points_count))+y_offset
    ring_Z = radius*np.sin(np.linspace(0, 2*np.pi, points_count))+z_offset
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

fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, phi, color='Blue')
ax_for_graphs.set_title('phi(t)')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, theta, color='Red')
ax_for_graphs.set_title('theta(t)')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, ROZ, color='Green')
ax_for_graphs.set_title('ROZ(t)')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(t, RB, color='Brown')
ax_for_graphs.set_title('RB(t)')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)


ring_coords1 = DrawRing(R_draw)
ring_coords2 = DrawRing(R_draw+2*r_draw)
point_coords = DrawRing(r_draw, z_offset=-R_draw-r_draw)

fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d([-9.0, 9.0])
ax.set_ylim3d([-9.0, 9.0])
ax.set_zlim3d([-7.0, 7.0])

ring1, = ax.plot(*ring_coords1, c='blue')
ring2, = ax.plot(*ring_coords2, c='blue')
point, = ax.plot(*point_coords, c="red")

ax.plot([0, 0], [0, 0], [R_draw + 2*r_draw, R_draw + r_draw + 5], color="blue")
ax.plot([0, 0], [0, 0], [-R_draw - 2*r_draw, -(R_draw + r_draw + 5)], color="blue")

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