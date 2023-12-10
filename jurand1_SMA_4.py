import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
g = 9.8  # Gravity (m/s^2)
m1 = 0.2  # Mass of object 1 (kg)
m2 = 0.4  # Mass of object 2 (kg)
v0 = 80  # Initial velocity (m/s)
ks = 0.015  # Drag coefficient when joined (kg/m)
t_separate = 1  # Time when the objects begin to move separately (s)
k1 = 0.02  # Drag coefficient for object 1 (kg/m)
k2 = 0.005  # Drag coefficient for object 2 (kg/m)
t_max = 15  # Maximum time for the simulation (s)


# Unified motion function
def equations(t, y):
    if t < t_separate:
        dv_dt = -g - ks * y[0] ** 2 * np.sign(y[0]) / (m1 + m2)
        return [dv_dt, dv_dt]
    else:
        dv1_dt = -g - k1 * y[0] ** 2 * np.sign(y[0]) / m1
        dv2_dt = -g - k2 * y[1] ** 2 * np.sign(y[1]) / m2
        return [dv1_dt, dv2_dt]


# Euler method
def euler_method(f, t_span, y0, steps):
    t0, tf = t_span
    h = (tf - t0) / steps
    t_values = np.linspace(t0, tf, steps + 1)
    y_values = np.zeros((2, steps + 1))
    y_values[:, 0] = y0
    for i in range(steps):
        a = f(t_values[i], y_values[:, i])
        y_values[0, i + 1] = y_values[0, i] + h * a[0]
        y_values[1, i + 1] = y_values[1, i] + h * a[1]
    return t_values, y_values


# Runge-Kutta 4th order method
def runge_kutta_4th_order(f, t_span, y0, steps):
    t0, tf = t_span
    h = (tf - t0) / steps
    t_values = np.linspace(t0, tf, steps + 1)
    y_values = np.zeros((2, steps + 1))
    y_values[:, 0] = y0
    for i in range(steps):
        k1 = np.array(f(t_values[i], y_values[:, i]))
        k2 = np.array(f(t_values[i] + h / 2, y_values[:, i] + h / 2 * k1))
        k3 = np.array(f(t_values[i] + h / 2, y_values[:, i] + h / 2 * k2))
        k4 = np.array(f(t_values[i] + h, y_values[:, i] + h * k3))
        y_values[:, i + 1] = y_values[:, i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t_values, y_values


# Initial conditions
initial_conditions = [v0, v0]

# Time points
t_points = np.linspace(0, t_max, 1000)

# Solve using solve_ivp with corrected event to stop at peak
sol = solve_ivp(equations, [0, t_max], initial_conditions, t_eval=t_points)

# Solving using Euler method
t_euler, y_euler = euler_method(equations, [0, t_max], initial_conditions, 1000)

# Solving using Runge-Kutta 4th order method
t_rk4, y_rk4 = runge_kutta_4th_order(equations, [0, t_max], initial_conditions, 1000)

# Plotting the results
plt.figure(figsize=(11, 8))

# Velocity vs Time
plt.plot(
    t_euler, y_euler[0], label="Object 1 Velocity (Euler)", color="#1f77b4"
)  # Blue color
plt.plot(
    t_euler,
    y_euler[1],
    label="Object 2 Velocity (Euler)",
    color="#1f77b4",
    linestyle=":",
)  # Dotted blue

plt.plot(
    t_rk4, y_rk4[0], label="Object 1 Velocity (RK4)", color="#ff7f0e"
)  # Orange color
plt.plot(
    t_rk4, y_rk4[1], label="Object 2 Velocity (RK4)", color="#ff7f0e", linestyle=":"
)  # Dotted orange

plt.plot(
    sol.t, sol.y[0], label="Object 1 Velocity (solve_ivp)", color="#2ca02c"
)  # Green color
plt.plot(
    sol.t,
    sol.y[1],
    label="Object 2 Velocity (solve_ivp)",
    color="#2ca02c",
    linestyle=":",
)  # Dotted green

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Find zero velocity points
zero_vel1 = np.argmin(np.abs(sol.y[0]))
zero_vel2 = np.argmin(np.abs(sol.y[1]))


def test_accuracy(
    method, func, time_span, initial_conditions, step_sizes, reference_solution
):
    colors = [
        "#0B0B0B",
        "#2F4F4F",
        "#483D8B",
        "#800000",
        "#8B0000",
        "#A52A2A",
        "#DC143C",
        "#FF4500",
        "#FF6347",
        "#FFA07A",
    ]
    accuracies = []
    i = 0
    for steps in step_sizes:
        num_steps = int(t_max / steps)
        t, y = method(func, time_span, initial_conditions, num_steps)
        # Calculate the absolute difference at the final point
        accuracy = np.abs(y[:, -1] - reference_solution[:, -1])
        accuracies.append(accuracy)

        plt.plot(
            t,
            y[0],
            label=f"Pirmas objektas žingsniu {steps}",
            color=colors[i],
        )
        plt.plot(
            t,
            y[1],
            label=f"Antras objektas žingsniu {steps}",
            linestyle="--",
            color=colors[i],
        )

        # Find zero velocity points
        zero_vel_index_0 = np.argmin(np.abs(y[0]))
        zero_vel_index_1 = np.argmin(np.abs(y[1]))

        # Print the time values
        print(
            f"1 objektas pasiekia aukščiausią tašką (žingsnis {steps}) laiku {t[zero_vel_index_0]}s"
        )
        print(
            f"2 objektas pasiekia aukščiausią tašką (žingsnis {steps}) laiku {t[zero_vel_index_1]}s"
        )

        i += 1

    plt.plot(sol.t, sol.y[0], label="Pirmo objekto teisingas ats", color="green")
    plt.plot(sol.t, sol.y[1], label="Antro objekto teisingas ats", color="blue")

    print(f"1 objektas pasiekia aukščiausią tašką (tikslus) laiku {sol.t[zero_vel1]}s")
    print(f"2 objektas pasiekia aukščiausią tašką (tikslus) laiku {sol.t[zero_vel2]}s")
    # Plot settings
    plt.xlabel("Laikas (s)")
    plt.ylabel("Greitis (m/s)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return accuracies


# Define a range of step sizes to test
step_sizes = [0.8, 0.6, 0.4, 0.2, 0.1, 0.01]

# Reference solution using solve_ivp with a high number of points
ref_sol = solve_ivp(
    equations, [0, t_max], initial_conditions, t_eval=np.linspace(0, t_max, 10000)
)


# Testing the Euler method
plt.figure(figsize=(11, 8))
plt.title("Greitis pagal laiką (Euleris)")
euler_accuracies = test_accuracy(
    euler_method, equations, [0, t_max], initial_conditions, step_sizes, ref_sol.y
)

# Testing the Runge-Kutta 4th order method
plt.figure(figsize=(11, 8))
plt.title("Greitis pagal laiką (Runge)")
rk4_accuracies = test_accuracy(
    runge_kutta_4th_order,
    equations,
    [0, t_max],
    initial_conditions,
    step_sizes,
    ref_sol.y,
)

euler_accuracies_plot = [np.linalg.norm(acc) for acc in euler_accuracies]
rk4_accuracies_plot = [np.linalg.norm(acc) for acc in rk4_accuracies]

# Plotting the corrected results
plt.figure(figsize=(11, 8))

# Accuracy vs Step size
plt.plot(
    step_sizes, euler_accuracies, label="Euler", marker="o", color="#d62728"
)  # Red color
plt.plot(
    step_sizes,
    rk4_accuracies,
    label="Runge-Kutta 4th Order",
    marker="^",
    color="#9467bd",
)  # Purple color

plt.xlabel("Step Size")
plt.ylabel("Norm of Absolute Difference at Final Time Point")
plt.title("Accuracy Comparison of Numerical Methods")
plt.legend()
plt.grid(True)
plt.show()
