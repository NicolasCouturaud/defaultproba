import pandas as pd
import numpy as np

# Define parameters
V0 = 100  # Replace with the desired initial value
r = 0.05  # Replace with the desired growth rate
sigmaA = 0.1  # Replace with the desired volatility
horizon = 10  # Replace with the desired size

# Generate Vt values for each time step (t from 1 to 10)
Vt = [np.mean(np.exp((r - sigmaA ** 2 / 2) * t + sigmaA * np.random.normal(0, np.sqrt(t), size=(1, horizon)).T)) for t in range(1, horizon + 1)]

# Create a DataFrame with one row and columns labeled with time steps
V = pd.DataFrame([Vt], columns=[f"t = {i}" for i in range(1, horizon + 1)])

# Display the DataFrame
print(V)
