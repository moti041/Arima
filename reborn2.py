import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd
MV=20*1E3
xt = 0

# Define the Ricker wavelet function
def ricker_wavelet(t, f):
    """Generate Ricker wavelet for given central frequency f."""
    pi = np.pi
    a = 1- 2*(pi*f*t)**2
    return a * np.exp(-(pi*f*t)**2)
def t_delay(xs, ys, xr):
    min_v=1500
    m1 = -ys/(xt-xs)
    m2 = -ys / (xr - xs)
    sq1, sq2 = np.sqrt(1+ m1**-2), np.sqrt(1+ m2**-2)
    return (1/MV)* (sq1+sq2)*np.log((MV*ys+min_v)/min_v)
def t_delay_snell( ys_v,vy):
    df = pd.DataFrame(columns=["x", "tau1", 'teta1', 'ys'])
    dy = np.diff(ys_v)[1]
    for teta1 in np.arange(0.001,-0.01+pi/2,(pi/2)/90):
        for n in range(len(ys_v)-1): # y val of reflect
            x, tau1, teta_n = 0,0 ,teta1
            for i in range(n-1):
               v1,v2=vy[i,0], vy[i+1,0]
               dx = dy * np.tan(teta_n)
               r = np.sqrt(dy ** 2 + dx ** 2)

               if np.isnan( np.arcsin(v2*np.sin(teta_n)/v1)):
                   break # critic angle : total reflection
               else:
                   teta_n= np.arcsin(v2*np.sin(teta_n)/v1)
               tau1+= r/v1
               x+=dx
            if ~np.isnan(teta_n):
                df = df._append({"x": x, "tau1": tau1, 'teta1':teta1, 'ys':ys_v[n]}, ignore_index=True)
    return df
def find_tau(df,xi,yi):
    # Calculate the absolute differences
    df["x_diff"] = abs(df["x"] - xi)
    df["y_diff"] = abs(df["ys"] - yi)
    # Calculate the total distance
    df["total_distance"] = df["x_diff"]**2 + df["y_diff"]**2
    # Find the row with the smallest total distance
    closest_row = df.loc[df["total_distance"].idxmin()]
    # Extract the tau1 value from the closest row
    return closest_row["tau1"]

    min_v=1500
f = 1E6  # Example frequency
# Generate meshgrids for xs and ys
xs_v = np.linspace(1E-8, 110*1E-3, 50)
ys_v = np.linspace(1E-8, 125*1E-3, 50)
xs, ys = np.meshgrid(xs_v, ys_v)
vy=MV*ys + 1500
propagate_vars = t_delay_snell( ys_v,vy)

# Define intervals in xs and ys where vy should be set to 0
xs_interval = (48.5E-3, 51.5E-3)  # Example interval for xs
ys_interval = xs_interval  # Example interval for ys

# Apply the condition to set vy to 0 in the specified intervals
condition = (xs >= xs_interval[0]) & (xs <= xs_interval[1]) & (ys >= ys_interval[0]) & (ys <= ys_interval[1])
vy[condition] = 0.01 # to prevent Nan is only almost 0
# Define the time variable t
t = np.linspace(-8E-5, 8E-5, 100)+5.8154210852011665e-05

# Calculate t_delay for each (xs, ys)
xr_v = [k*10E-3 for k in range(2,9)]
plt.figure(figsize=(10, 6))
for xr in xr_v:
    delay = t_delay(xs, ys, xr)
    # Evaluate the Ricker wavelet at t0 = t - t_delay
    # Initialize an array to store the Ricker wavelet values
    wavelet_values = np.zeros(( len(ys), len(t)))
    # Central frequency of the Ricker wavelet
    tr=1 # transmit
    for j in range(len(ys)-1):
        tr *= 4 * vy[ j ,0] * vy[ j + 1,0]/ ((vy[ j,0] + vy[ j + 1,0])**2) # multiplication of tx_coef at 2 ways
        xs =(xr+xt)/2 # due to specular reflection
        # mask = (xs >= xs_interval[0]) & (xs <= xs_interval[1]) & (ys[j,0] >= ys_interval[0]) & (
        #             ys[ j,0] <= ys_interval[1])
        tr_mask =  tr
        re = np.abs((vy[j + 1, 0] - vy[j, 0]) / (vy[j, 0] + vy[j + 1, 0])) # reflection coeff
        t0 = t - 2*find_tau(propagate_vars,xs,ys[j,0]) # multiply by 2 due to specular reflection
        # t0 -= np.mean(t0)
        wavelet_values[j, :] = (tr_mask ** 1) * re * ricker_wavelet(t0, f)  #
        # for i in range(len(xs)):
        #     mask = (xs[i, j] >= xs_interval[0]) & (xs[i, j] <= xs_interval[1]) & (ys[i, j] >= ys_interval[0]) & (ys[i, j] <= ys_interval[1])
        #     tr_mask = 0 if mask else tr
        #     re = np.abs((vy[ j+1,i] - vy[ j,i])/ (vy[ j,i] + vy[ j + 1,i]))
        #     t0 = t - delay[ j,i]
        #    # t0 -= np.mean(t0)
        #     wavelet_values[j, :] = (tr_mask**1) * re * ricker_wavelet(t0, f) #
    # Sum wavelet values over xs and ys for each t value
    summed_wavelet = np.sum(wavelet_values, axis=(0))
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
