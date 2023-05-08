import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#constantes
mass_1 = 4
mass_2 = 5
g = 9.81
length = 10

h = 0.01
t_max = 10
n =int(t_max/h) + 1
w_0 = np.sqrt(g/length)


#lista con valores
thetas_1 = []
omegas_1 = []
thetas_2 = []
omegas_2 = []

def f1(theta1,omega1,theta2,omega2):
    numerador = -g * (2 * mass_1 + mass_2) * np.sin(theta1) - mass_2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * mass_2 * (omega2**2 * length + omega1**2 * length * np.cos(theta1 - theta2))
    denominador = length * (2 * mass_1 + mass_2 - mass_2 * np.cos(2 * theta1 - 2 * theta2))
    return numerador / denominador

def f2(theta1, omega1, theta2, omega2):
    numerador = 2 * np.sin(theta1 - theta2) * (omega1**2 * length * (mass_1 + mass_2) + g * (mass_1 + mass_2) * np.cos(theta1) + omega2**2 * length * mass_2 * np.cos(theta1 - theta2))
    denominador = length * (2 * mass_1 + mass_2 - mass_2 * np.cos(2 * theta1 - 2 * theta2))
    return numerador / denominador

def rk4():
    #valores iniciales
    theta_0 = math.radians(60)
    omega_0 = 0
    omegas_1.append(omega_0)
    thetas_1.append(theta_0)
    omegas_2.append(omega_0)
    thetas_2.append(theta_0)
    for i in range(1,n):
    #metodo runge_kutta parte 1
        k_1 = h * f1(thetas_1[i-1],omegas_1[i-1],thetas_2[i-1],omegas_2[i-1])
        c = 0.5 * k_1 * h
        k_2 = h * f1(thetas_1[i-1] + c,omegas_1[i-1] + c,thetas_2[i-1] + c,omegas_2[i-1] + c)
        c = 0.5 * k_2 * h
        k_3 = h * f1(thetas_1[i-1] + c,omegas_1[i-1] + c,thetas_2[i-1] + c,omegas_2[i-1] + c)
        c =  k_3 * h
        k_4 = h * f1(thetas_1[i-1] + c,omegas_1[i-1] + c,thetas_2[i-1] + c,omegas_2[i-1] + c)

        angle = omegas_1[i-1] + ((k_1 + (2 * k_2) + (2 * k_3) + k_4) * (1/6))
        omegas_1.append(angle)
        

        k_1 = h * omegas_1[i-1]
        k_2 = h * (omegas_1[i-1] + k_1* 0.5)
        k_3 =  h * (omegas_1[i-1] + k_2* 0.5)
        k_4 = h * (omegas_1[i-1] + k_3)

        angle = thetas_1[i-1] + ((k_1 + (2 * k_2) + (2 * k_3) + k_4) * (1/6.0))
        thetas_1.append(angle)

        #runge_kutta parte 2
        k_1 = h * f2(thetas_1[i-1],omegas_1[i-1],thetas_2[i-1],omegas_2[i-1])
        c = 0.5 * k_1 * h
        k_2 = h * f2(thetas_1[i-1] + c,omegas_1[i-1] + c,thetas_2[i-1] + c,omegas_2[i-1] + c)
        c = 0.5 * k_2 * h
        k_3 = h * f2(thetas_1[i-1] + c,omegas_1[i-1] + c,thetas_2[i-1] + c,omegas_2[i-1] + c)
        c =  k_3 * h
        k_4 = h * f2(thetas_1[i-1] + c,omegas_1[i-1] + c,thetas_2[i-1] + c,omegas_2[i-1] + c)

        angle = omegas_2[i-1] + ((k_1 + (2 * k_2) + (2 * k_3) + k_4) * (1/6))
        omegas_2.append(angle)
        

        k_1 = h * omegas_2[i-1]
        k_2 = h * (omegas_2[i-1] + k_1* 0.5)
        k_3 =  h * (omegas_2[i-1] + k_2* 0.5)
        k_4 = h * (omegas_2[i-1] + k_3)

        angle = thetas_2[i-1] + ((k_1 + (2 * k_2) + (2 * k_3) + k_4) * (1/6.0))
        thetas_2.append(angle)

    return thetas_1,thetas_2
angle = rk4()
# animacion
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(1, 1, 1)  # inicializar la figura y el eje
ax.set_xlim(-3*length, 3*length)
ax.set_ylim(-3*length, 3*length)
ax.set_aspect('equal')
# inicializar el objeto de línea para la soga
rope, = ax.plot([], [], 'k-', lw=1)
rope2, = ax.plot([], [], 'k-', lw=1)
# inicializar el objeto de línea para la masa
mass, = ax.plot([], [], 'o-', color='blue', lw=2)
mass2, = ax.plot([], [], 'o-', color='blue', lw=2)


def init():
    rope.set_data([], [])
    rope2.set_data([], [])
    mass.set_data([], [])
    mass2.set_data([], [])
    return rope, mass,rope2,mass2



def animate(i):
    # calcular la posición de la masa y el punto de pivote
    x_pivot = 0
    y_pivot = 0
    x_mass = length * np.sin(angle[0][i])
    y_mass = -length * np.cos(angle[0][i])


    x_mass2 = x_mass + length * math.sin(angle[1][i])
    y_mass2 = y_mass - length * math.cos(angle[1][i])
    rope.set_data([x_pivot, x_mass], [y_pivot, y_mass])  # trazar la soga
    rope2.set_data([x_mass, x_mass2], [y_mass, y_mass2])
    mass.set_data(x_mass, y_mass)  # trazar la masa
    mass2.set_data(x_mass2, y_mass2)
    return rope, mass,rope2,mass2


anim = FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=5, blit=True)
plt.show()

