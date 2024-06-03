import numpy as np
import matplotlib.pyplot as plt
MV=20*1E3
# Define the Ricker wavelet function
def ricker_wavelet(t, f):
    """Generate Ricker wavelet for given central frequency f."""
    a = 2 / (np.sqrt(3 * f) * (np.pi ** 0.25))
    return a * (1 - (t ** 2) * (f ** 2)) * np.exp(-(t ** 2) * (f ** 2) / 2)

def t_delay(xs, ys, xr):
    xt=0
    min_v=1500
    m1 = -ys/(xt-xs)
    m2 = -ys / (xr - xs)
    sq1, sq2 = np.sqrt(1+ m1**-2), np.sqrt(1+ m2**-2)
    return (1/MV)* (sq1+sq2)*np.log((MV*ys+min_v)/min_v)

# Generate meshgrids for xs and ys
xs = np.linspace(1E-8, 110*1E-3, 50)
ys = np.linspace(1E-8, 125*1E-3, 50)
xs, ys = np.meshgrid(xs, ys)
vy=MV*ys + 1500
# Define intervals in xs and ys where vy should be set to 0
xs_interval = (48E-3, 52E-3)  # Example interval for xs
ys_interval = xs_interval  # Example interval for ys

# Apply the condition to set vy to 0 in the specified intervals
condition = (xs >= xs_interval[0]) & (xs <= xs_interval[1]) & (ys >= ys_interval[0]) & (ys <= ys_interval[1])
vy[condition] = 0
# Define the time variable t
t = np.linspace(-1, 1, 100)

# Calculate t_delay for each (xs, ys)
xr = 5*1E-3
delay = t_delay(xs, ys, xr)
# Evaluate the Ricker wavelet at t0 = t - t_delay
# Initialize an array to store the Ricker wavelet values
wavelet_values = np.zeros((len(xs), len(ys), len(t)))
# Central frequency of the Ricker wavelet
f = 1E6  # Example frequency
tr=1 # transmit
for j in range(len(ys)-1):
    tr *= 2 * vy[0, j + 1] / (vy[0, j] + vy[0, j + 1])
    for i in range(len(xs)):
        mask = (xs[i, j] >= xs_interval[0]) & (xs[i, j] <= xs_interval[1]) & (ys[i, j] >= ys_interval[0]) & (ys[i, j] <= ys_interval[1])
        tr_mask = 0 if mask else tr
        re = np.abs((vy[i, j+1] - vy[i, j])/ (vy[i, j] + vy[i, j + 1]))
        t0 = t - delay[i, j]
        wavelet_values[i, j, :] = ricker_wavelet(t0, f) #

# Plotting a single Ricker wavelet for visualization
plt.figure(figsize=(10, 6))
plt.plot(t, ricker_wavelet(t, f), label='Original Ricker Wavelet')
plt.plot(t, ricker_wavelet(t - delay[50, 50], f), label=f'Ricker Wavelet with t_delay at (50, 50)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Ricker Wavelet')
plt.show()

# Example: Visualizing the delay field
plt.figure(figsize=(10, 6))
plt.contourf(xs, ys, delay, cmap='viridis')
plt.colorbar(label='t_delay')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('t_delay over the plane')
plt.show()
