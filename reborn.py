import numpy as np
import matplotlib.pyplot as plt
MV=20*1E3
# Define the Ricker wavelet function
def ricker_wavelet(t, f):
    """Generate Ricker wavelet for given central frequency f."""
    pi = np.pi
    a = 1- 2*(pi*f*t)**2
    return a * np.exp(-(pi*f*t)**2)
def t_delay(xs, ys, xr):
    xt=0
    min_v=1500
    m1 = -ys/(xt-xs)
    m2 = -ys / (xr - xs)
    sq1, sq2 = np.sqrt(1+ m1**-2), np.sqrt(1+ m2**-2)
    return (1/MV)* (sq1+sq2)*np.log((MV*ys+min_v)/min_v)

f = 1E6  # Example frequency
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
vy[condition] = 0.01 # to prevent Nan is only almost 0
# Define the time variable t
t = np.linspace(-2E-5, 2E-5, 100)+5.8154210852011665e-05

# Calculate t_delay for each (xs, ys)
xr_v = [k*10E-3 for k in range(2,9)]
plt.figure(figsize=(10, 6))
for xr in xr_v:
    delay = t_delay(xs, ys, xr)
    # Evaluate the Ricker wavelet at t0 = t - t_delay
    # Initialize an array to store the Ricker wavelet values
    wavelet_values = np.zeros((len(xs), len(ys), len(t)))
    # Central frequency of the Ricker wavelet
    tr=1 # transmit
    for j in range(len(ys)-1):
        tr *= 2 * vy[ j + 1,0] / (vy[ j,0] + vy[ j + 1,0])
        for i in range(len(xs)):
            mask = (xs[i, j] >= xs_interval[0]) & (xs[i, j] <= xs_interval[1]) & (ys[i, j] >= ys_interval[0]) & (ys[i, j] <= ys_interval[1])
            tr_mask = 0 if mask else tr
            re = np.abs((vy[ j+1,i] - vy[ j,i])/ (vy[ j,i] + vy[ j + 1,i]))
            t0 = t - delay[ j,i]
           # t0 -= np.mean(t0)
            wavelet_values[i, j, :] = (tr_mask**2) * re * ricker_wavelet(t0, f) #
    # Sum wavelet values over xs and ys for each t value
    summed_wavelet = np.sum(wavelet_values, axis=(0, 1))
# Plotting a single Ricker wavelet for visualization
    plt.plot(t,summed_wavelet, label=xr)
plt.legend()
plt.show()
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
