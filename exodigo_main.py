import numpy as np
from scipy.signal import find_peaks
# Load the .npy file
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# Define a function to fit a parabola to a neighborhood of points
def parabola(x, a, b, c):
    return a * x**2 + b * x + c
# Function to calculate the goodness of fit (e.g., R-squared)
def goodness_of_fit(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared
# Define a function to fit a parabola and return the parameters and goodness of fit
def fit_parabola(x, y):
    popt, _ = curve_fit(parabola, x, y)
    y_fit = parabola(x, *popt)
    r_squared = goodness_of_fit(y, y_fit)
    return popt, r_squared
idx = 2000
data_all = np.load('gps_path.npy')
data_with_noise = data_all[:idx,:]#TODO
# No smoothing is applied directly
# Smooth the path using moving average or any other suitable method
win = 10
smoothed_y = np.convolve(data_with_noise[:, 1], np.ones(win)/win, mode='valid')  # Adjust the window size as needed
smoothed_x = np.convolve(data_with_noise[:, 0], np.ones(win)/win, mode='valid')
data = np.column_stack((smoothed_x, smoothed_y))

# Find local maxima and minima indices
max_indices = argrelextrema(data[:,1], np.greater)[0]
min_indices = argrelextrema(data[:,1], np.less)[0]

# Threshold for goodness of fit
threshold = 0.5  # You need to adjust this threshold based on your data and requirements
max_real, min_real =[],[]
# Verify alternation between maxima and minima
for i in range(min(len(max_indices), len(min_indices))):
    max_index = max_indices[i]
    min_index = min_indices[i]
    width=2
    # Define neighborhoods around maxima and minima
    max_start_index = max(0, max_index - width)  # Adjust the window size as needed
    max_end_index = min(len(data) - 1, max_index + width)  # Adjust the window size as needed
    min_start_index = max(0, min_index - width)  # Adjust the window size as needed
    min_end_index = min(len(data) - 1, min_index + width)  # Adjust the window size as needed

    # Fit parabolas to neighborhoods
    max_neighborhood_x = data[max_start_index:max_end_index, 0]
    max_neighborhood_y = data[max_start_index:max_end_index, 1]
    min_neighborhood_x = data[min_start_index:min_end_index, 0]
    min_neighborhood_y = data[min_start_index:min_end_index, 1]

    max_popt, max_r_squared = fit_parabola(max_neighborhood_x, max_neighborhood_y)
    min_popt, min_r_squared = fit_parabola(min_neighborhood_x, min_neighborhood_y)

    # Check if the fits are good and alternate between maxima and minima
    if max_r_squared > threshold:# and min_r_squared > threshold :#TODO and i % 2 == 0:
        max_real.append(max_index)
    if min_r_squared > threshold:  # and min_r_squared > threshold :#TODO and i % 2 == 0:
        min_real.append(min_index)
        # print("Genuine Maximum at x={}, y={}, R-squared={}".format(data[max_index, 0], data[max_index, 1], max_r_squared))
        # print("Genuine Minimum at x={}, y={}, R-squared={}".format(data[min_index, 0], data[min_index, 1], min_r_squared))
plt.figure()
# Plot the original data and the identified maxima and minima
plt.plot(data[:idx, 0], data[:idx, 1], label='Original data')
plt.plot(data[:, 0][max_real], data[:, 1][max_real], 'ro', label='Maxima')
plt.plot(data[:, 0][min_real], data[:, 1][min_real], 'go', label='Maxima')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original data with Extrema')
plt.show()
pass