import numpy as np
import matplotlib.pyplot as plt

def ssfp_fid(M_0, theta, E_1, E_2):
    # Convert theta from degrees to radians 
    theta_rad = np.radians(theta) 
    
    # Compute p and q based on the new definitions
    p = 1 - E_1 * np.cos(theta_rad) - E_2**2 * (E_1 - np.cos(theta_rad))
    q = E_2 * (1 - E_1) * (1 + np.cos(theta_rad))
    
    # Compute the SSFP_FID expression
    result = (M_0 * np.tan(theta_rad / 2) * 
              (1 - ((E_1 - np.cos(theta_rad)) * (1 - E_2**2)) / np.sqrt(p**2 - q**2)))
    
    return result

def calcSteadyStateMean(signal, gradient_threshold, plot = False, print_info = False):
    """
    Calculates the mean of the 'flat' part of the MRI signal.
    The 'flatness' of the curve is determined by the differential.
    """
    # ensure that the sequence is flattened
    signal = signal.flatten()
    
    # calculate the numerical gradient of the sequence
    gradient = np.gradient(signal, 1) # spacing between values is 1

    # find indices where the gradient is below the steady state threshold
    steadystate_indices = np.where(np.abs(gradient) < gradient_threshold)[0]

    # notify if there is no steady state found
    if len(steadystate_indices) == 0:
        raise ValueError("No steady state found based on the gradient threshold.")
    

    # extract steady state
    steadystate_values = signal[steadystate_indices[0]:steadystate_indices[-1]+1] # plus one to include the edge case

    # Calculate the average of the flat region
    steadystate_avg = np.mean(steadystate_values)

    if plot == True:
        plt.plot(signal, label='Signal')
        plt.axvline(x=steadystate_indices[0], color='r', linestyle='--', label='Start of Steady State')
        plt.axvline(x=steadystate_indices[-1]+1, color='b', linestyle='--', label='End of Steady State')
        plt.legend()
        plt.xlabel('Dynamics')
        plt.ylabel('Signal (AU)')
        plt.title('MR Signal with Steady State Region Calculated')
        plt.show()

    if print_info == True:
        print(f"Steady-state starts at index {steadystate_indices[0]} and ends at index {steadystate_indices[-1]+1}")
        print(f"Average value of steady state: {steadystate_avg:.2f}")

    return steadystate_avg
