import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# constantes
mass = 4
g = 9.81
length = 10
h = 0.001
t_max = 200
n = int(t_max/h) + 1
w_0 = np.sqrt(g/length)
omega_0 = 0
theta_0 = math.radians(3)


def energy(theta, omega):
    k = (mass * (length**2) * (omega**2) * 0.5)
    p = ((mass * g * length) - (mass * g * length * np.cos(theta)))

    return (k+p, k, p)


def ftheta(theta):
    return (-(w_0**2) * (np.sin(theta)))


def eu_explicito():
    thetas = []
    omegas = []
    T_Energy = []
    K_Energy = []
    P_Energy = []
    omegas.append(omega_0)
    thetas.append(theta_0)
    T_Energy.append(energy(theta_0, omega_0)[0])
    K_Energy.append(energy(theta_0, omega_0)[1])
    P_Energy.append(energy(theta_0, omega_0)[2])
    # metodo euler explicito
    for i in range(1, n):
        thetas.append(thetas[i-1] + h * omegas[i-1])
        omegas.append(omegas[i-1] + h * ftheta(thetas[i-1]))
        T_Energy.append(energy(thetas[i], omegas[i])[0])
        K_Energy.append(energy(thetas[i], omegas[i])[1])
        P_Energy.append(energy(thetas[i], omegas[i])[2])

    # plotear grafico
    '''
    time = np.linspace(0, t_max, n)
    plt.plot(time, T_Energy)
    plt.plot(time, K_Energy)
    plt.plot(time, P_Energy)
    plt.title('Energia sobre el tiempo, Euler explicito')
    plt.xlabel('tiempo')
    plt.ylabel('energia')
    plt.legend()
    plt.show()'''
    return thetas


def eu_semi_expl():
    thetas = []
    omegas = []
    T_Energy = []
    K_Energy = []
    P_Energy = []
    omegas.append(omega_0)
    thetas.append(theta_0)
    T_Energy.append(energy(theta_0, omega_0)[0])
    K_Energy.append(energy(theta_0, omega_0)[1])
    P_Energy.append(energy(theta_0, omega_0)[2])
    # metodo euler explicito
    for i in range(1, n):
        omegas.append(omegas[i-1] + h * ftheta(thetas[i-1]))
        thetas.append(thetas[i-1] + h * omegas[i])

        T_Energy.append(energy(thetas[i], omegas[i])[0])
        K_Energy.append(energy(thetas[i], omegas[i])[1])
        P_Energy.append(energy(thetas[i], omegas[i])[2])
        # plotear grafico
    '''time = np.linspace(0, t_max, n)
    plt.plot(time, T_Energy)
    plt.plot(time, K_Energy)
    plt.plot(time, P_Energy)
    plt.title('Energia sobre el tiempo, Euler-semi-explicito')
    plt.xlabel('tiempo')
    plt.ylabel('energia')
    plt.legend()
    plt.show()'''
    return thetas


def rungekutta_4():
    thetas = []
    omegas = []
    T_Energy = []
    K_Energy = []
    P_Energy = []
    omegas.append(omega_0)
    thetas.append(theta_0)
    T_Energy.append(energy(theta_0, omega_0)[0])
    K_Energy.append(energy(theta_0, omega_0)[1])
    P_Energy.append(energy(theta_0, omega_0)[2])

    for i in range(1, n):
        # metodo runge_kutta
        k_1 = h * ftheta(thetas[i-1])
        k_2 = h * ftheta(thetas[i-1] + 0.5 * k_1 * h)
        k_3 = h * ftheta(thetas[i-1] + 0.5 * k_2 * h)
        k_4 = h * ftheta(thetas[i-1] + k_3 * h)

        angle = omegas[i-1] + ((k_1 + (2 * k_2) + (2 * k_3) + k_4) * (1/6))
        omegas.append(angle)

        k_1 = h * omegas[i-1]
        k_2 = h * (omegas[i-1] + k_1 * 0.5)
        k_3 = h * (omegas[i-1] + k_2 * 0.5)
        k_4 = h * (omegas[i-1] + k_3)

        angle = thetas[i-1] + ((k_1 + (2 * k_2) + (2 * k_3) + k_4) * (1/6.0))
        thetas.append(angle)

        # calculo de energia
        T_Energy.append(energy(thetas[i], omegas[i])[0])
        K_Energy.append(energy(thetas[i], omegas[i])[1])
        P_Energy.append(energy(thetas[i], omegas[i])[2])
    '''time = np.linspace(0, t_max, n)
    plt.plot(time, T_Energy)
    plt.plot(time, K_Energy)
    plt.plot(time, P_Energy)
    plt.title('Energia sobre el tiempo, rungekutta 4')
    plt.xlabel('tiempo')
    plt.ylabel('energia')
    plt.legend()
    plt.show()'''
    return thetas


# eu_semi_expl()
# angle = rungekutta_4()
'''
    #plotear grafico
    time = np.linspace(0,t_max,n)
    plt.plot(time, T_Energy)
    plt.plot(time, K_Energy)
    plt.plot(time, P_Energy)
    plt.title('Energy over time semi_expl-euler')
    plt.legend()
    plt.show()
'''

# animacion
'''fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(1, 1, 1)  # inicializar la figura y el eje
ax.set_xlim(-1.5*length, 1.5*length)
ax.set_ylim(-1.5*length, 1.5*length)
ax.set_aspect('equal')
# inicializar el objeto de línea para la soga
rope, = ax.plot([], [], 'k-', lw=1)
# inicializar el objeto de línea para la masa
mass, = ax.plot([], [], 'o-', color='blue', lw=2)'''


def init():
    rope.set_data([], [])
    mass.set_data([], [])
    return rope, mass,


def animate(i):
    # calcular la posición de la masa y el punto de pivote
    x_pivot = 0
    y_pivot = 0
    x_mass = length * np.sin(angle[i])
    y_mass = -length * np.cos(angle[i])
    rope.set_data([x_pivot, x_mass], [y_pivot, y_mass])  # trazar la soga
    mass.set_data(x_mass, y_mass)  # trazar la masa
    return rope, mass,


'''anim = FuncAnimation(fig, animate, init_func=init,
                     frames=10000, interval=1, blit=True)
plt.show()'''

'''
# inicializar la figura y el eje

fig, ax = plt.subplots()
ax.set_xlim(-1.5*length, 1.5*length)
ax.set_ylim(-1.5*length, 1.5*length)

# inicializar el objeto de línea
line, = ax.plot([], [], 'o-', lw=2)


def init():
    line.set_data([], [])
    return line,

angle = rungekutta_4()

#funcion que devuelve la posicion por cada frame
def animate(i):
    x = length * np.sin(angle[i])
    y = -length * np.cos(angle[i])
    line.set_data(x, y)
    return line,


anim = FuncAnimation(fig, animate, init_func=init,
                    frames=10000, interval=1, blit=True)

#anim.save("penduloco.mp4", writer="ffmpeg", fps=30)
plt.show()


'''


def plot_trayectoria():
    time = np.linspace(0.0, t_max, n)
    theta = theta_0 * np.cos(w_0*time)
    theta2 = eu_explicito()
    theta3 = eu_semi_expl()
    theta4 = rungekutta_4()

    plt.plot(time, theta, label='Función analítica')
    plt.plot(time, theta2, label='Aproximación de euler')
    plt.title('Theta analítica vs Euler explícito')
    plt.xlabel('tiempo')
    plt.ylabel('theta')
    plt.legend(loc='lower left')
    plt.show()

    plt.plot(time, theta, label='Función analítica')
    plt.plot(time, theta3, label='Aproximación de euler semi-explícito')
    plt.title('Theta analítica vs theta euler semi-explícito')
    plt.xlabel('tiempo')
    plt.ylabel('theta')
    plt.legend(loc='lower left')
    plt.show()

    plt.plot(time, theta, label='Función analítica')
    plt.plot(time, theta4, label='Aproximación de RK4')
    plt.title('Theta analítica vs RK4')
    plt.xlabel('tiempo')
    plt.ylabel('theta')
    plt.legend(loc='lower left')
    plt.show()


plot_trayectoria()
