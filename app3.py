import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import time
import pandas as pd
import streamlit as st


with st.expander("App Information"):
    st.markdown("""
    ## Lorenz System

    The Lorenz system, first studied by Edward Lorenz, is a system of ordinary differential equations. It is known for its chaotic solutions under certain parameter values and initial conditions. The Lorenz attractor, a set of chaotic solutions of the Lorenz system, is particularly noteworthy. The 'butterfly effect' in popular media originates from the real-world implications of the Lorenz attractor.

    ### Variables in the Lorenz System

    - `x`, `y`, `z`: These variables represent the state of the system at any given time.
    - `sigma`: This is the Prandtl number, a parameter representing the ratio of momentum diffusivity to thermal diffusivity.
    - `rho`: This is the Rayleigh number, a parameter describing the flow regime in fluid dynamics.
    - `beta`: This is a geometric factor.

    ## Lyapunov Exponent

    The Lyapunov exponent is a measure that characterizes the rate of separation of infinitesimally close trajectories. A positive Lyapunov exponent indicates that a system exhibits chaotic dynamics. In the context of the Lorenz system, the Lyapunov exponent can be used to identify the presence of the Lorenz attractor.

    You can halt the app using the `Stop` button in the top right corner to view the current state of the system. You can also export the system data and the Lyapunov exponent for each system using the `Export Data and see Lyapunov Exponents` checkbox. After checking it, you can export the x, y, z data and view the Lyapunov exponent for each system. Remember to check the bottom of sidebar to see the `Select a system to display its Lyapunov exponent:` dropdown menu, which allows you to choose the system for which you want to see the Lyapunov exponents.     
    """)

# Define the Lorenz system with Jacobian
def lorenz_system_with_jacobian(Y, t, sigma, rho, beta):
    x, y, z, X, Y, Z, P, Q, R = Y
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    dX_dt = sigma * (Y - X) - dx_dt * P
    dY_dt = (rho - z) * X - x * X - Y - dx_dt * Q
    dZ_dt = y * X + x * Y - beta * Z - dx_dt * R
    return [dx_dt, dy_dt, dz_dt, dX_dt, dY_dt, dZ_dt, P, Q, R]

# Function to calculate Lyapunov exponent
def lyapunov_exponent(t, solution):
    return 1/(t + 1e-7) * np.log(np.sqrt(solution[-1, 3]**2 + solution[-1, 4]**2 + solution[-1, 5]**2) + 1e-7)

# Set up the sidebar
st.sidebar.title('Lorenz System Parameters')
st.sidebar.markdown("---")
# Define the time points for the simulation
num_points = st.sidebar.slider('Number of Points for Time', 100, 5000, 1000)
t = np.linspace(0, 10, num_points)
st.sidebar.markdown("---")
sigma = st.sidebar.slider('Sigma', 0.0, 50.0, 10.0)
rho = st.sidebar.slider('Rho', 0.0, 50.0, 28.0)
beta = st.sidebar.slider('Beta', 0.0, 50.0, 2.67)
st.sidebar.markdown("---")
x0_1 = st.sidebar.number_input('Initial x for 1st system', value=1.0)
y0_1 = st.sidebar.number_input('Initial y for 1st system', value=1.0)
z0_1 = st.sidebar.number_input('Initial z for 1st system', value=1.0)
st.sidebar.markdown("---")
x0_2 = st.sidebar.number_input('Initial x for 2nd system', value=1.0)
y0_2 = st.sidebar.number_input('Initial y for 2nd system', value=1.0)
z0_2 = st.sidebar.number_input('Initial z for 2nd system', value=1.0)
st.sidebar.markdown("---")



# Solve the Lorenz system with Jacobian
initial_conditions_1 = [x0_1, y0_1, z0_1, 0, 0, 0, 0, 0, 0]
initial_conditions_2 = [x0_2, y0_2, z0_2, 0, 0, 0, 0, 0, 0]
solution_1 = odeint(lorenz_system_with_jacobian, initial_conditions_1, t, args=(sigma, rho, beta))
solution_2 = odeint(lorenz_system_with_jacobian, initial_conditions_2, t, args=(sigma, rho, beta))

# Calculate Lyapunov exponents
lyapunov_exponent_1 = lyapunov_exponent(t, solution_1)
lyapunov_exponent_2 = lyapunov_exponent(t, solution_2)


# Create a 3D plot of the solutions
plot_slot = st.empty()

stop = st.checkbox('Export Data and see Lyopunov Exponents')

i = 0
while i < len(t) and not stop:
    fig = go.Figure(data=[
        go.Scatter3d(x=solution_1[:i, 0], y=solution_1[:i, 1], z=solution_1[:i, 2], mode='lines', line=dict(color='red'), name='System 1'),
        go.Scatter3d(x=solution_2[:i, 0], y=solution_2[:i, 1], z=solution_2[:i, 2], mode='lines', line=dict(color='blue'), name='System 2')
    ])

    fig.update_layout(scene=dict(xaxis=dict(range=[min(min(solution_1[:, 0]), min(solution_2[:, 0])), max(max(solution_1[:, 0]), max(solution_2[:, 0]))]),
                                 yaxis=dict(range=[min(min(solution_1[:, 1]), min(solution_2[:, 1])), max(max(solution_1[:, 1]), max(solution_2[:, 1]))]),
                                 zaxis=dict(range=[min(min(solution_1[:, 2]), min(solution_2[:, 2])), max(max(solution_1[:, 2]), max(solution_2[:, 2]))]),
                                 camera=dict(eye=dict(x=2*np.cos(i/10), y=2*np.sin(i/10), z=0.1))),
                     width=800, height=600)

    plot_slot.plotly_chart(fig)

    i += 1
    time.sleep(0.1)

# Create a dictionary to store the Lyapunov exponents
lyapunov_exponents = {'System 1': lyapunov_exponent_1, 'System 2': lyapunov_exponent_2}

# Create a dropdown menu in the sidebar for the user to select which Lyapunov exponent to display
selected_system = st.sidebar.selectbox('Select a system to display its Lyapunov exponent:', list(lyapunov_exponents.keys()))

if st.button('Export Data'):
    df_1 = pd.DataFrame(solution_1[:, :3], columns=['x_2', 'y_2', 'z_2'])
    df_2 = pd.DataFrame(solution_2[:, :3], columns=['x_2', 'y_2', 'z_2'])
    
    csv_1 = df_1.to_csv(index=False)
    csv_2 = df_2.to_csv(index=False)
    
    st.download_button(
        label="Download data for System 1",
        data=csv_1,
        file_name='lorenz_data_1.csv',
        mime='text/csv',
    )

    st.download_button(
        label="Download data for System 2",
        data=csv_2,
        file_name='lorenz_data_2.csv',
        mime='text/csv',
    )

    # Export Lyapunov exponents for System 1
    df_lyapunov_1 = pd.DataFrame(lyapunov_exponent_1.reshape(-1, 1))
    csv_lyapunov_1 = df_lyapunov_1.to_csv(index=False)

    st.download_button(
        label="Download Lyapunov exponents for System 1",
        data=csv_lyapunov_1,
        file_name='lyapunov_exponents_1.csv',
        mime='text/csv',
    )

    # Export Lyapunov exponents for System 2
    df_lyapunov_2 = pd.DataFrame(lyapunov_exponent_2.reshape(-1, 1))
    csv_lyapunov_2 = df_lyapunov_2.to_csv(index=False)

    st.download_button(
        label="Download Lyapunov exponents for System 2",
        data=csv_lyapunov_2,
        file_name='lyapunov_exponents_2.csv',
        mime='text/csv',
    )

# Display the selected Lyapunov exponent
st.write(f'Lyapunov exponent for {selected_system}: {lyapunov_exponents[selected_system]}')

