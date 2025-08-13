import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Set matplotlib fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Experimental data (time: minutes, fluorescence intensity: arbitrary units)
experimental_time_min = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])
experimental_fluorescence = np.array([30.00, 44.13, 66.96, 98.05, 121.05, 
                                     151.40, 170.53, 193.55, 219.08, 238.75, 263.77, 273.53])

# Convert to seconds (kinetic equations based on seconds)
experimental_time_sec = experimental_time_min * 60

# Initial concentrations (units: M)
L0 = 30e-9    # Initial ligand concentration
R0 = 800e-9   # Initial receptor concentration
S1_0 = 800e-9 # Initial S1 concentration

# Simplified kinetic model
def model(y, t, k1, k1r, k2, k2r, k3):
    """
    Define ODE system (simplified LRin kinetics)
    y: state variables [L, R, LR, LRin, S1, S1S2F]
    t: time (seconds)
    k1: association rate constant, k1r: dissociation rate constant
    k2: forward flip rate constant, k2r: reverse flip rate constant
    k3: catalytic rate constant
    """
    L, R, LR, LRin, S1, S1S2F = y
    
    # Equation 1: d[L]/dt = -k1*[L][R] + k1r*[LR]
    dL_dt = -k1 * L * R + k1r * LR
    
    # Equation 2: d[R]/dt = -k1*[L][R] + k1r*[LR]
    dR_dt = -k1 * L * R + k1r * LR
    
    # Equation 3: d[LR]/dt = k1*[L][R] - k1r*[LR] - k2*[LR] + k2r*[LRin]
    dLR_dt = k1 * L * R - k1r * LR - k2 * LR + k2r * LRin
    
    # Equation 4: d[LRin]/dt = k2*[LR] - k2r*[LRin] 
    dLRin_dt = k2 * LR - k2r * LRin
    
    # Equation 5: d[S1]/dt = -k3*[LRin]*[S1] (S1 catalytic consumption)
    dS1_dt = -k3 * LRin * S1
    
    # Equation 6: d[S1S2F]/dt = k3*[LRin]*[S1] (fluorescent product formation)
    dS1S2F_dt = k3 * LRin * S1
    
    return [dL_dt, dR_dt, dLR_dt, dLRin_dt, dS1_dt, dS1S2F_dt]

# Wrapper function for curve fitting
def fluorescence_curve(t_sec, k1, k1r, k2, k2r, k3, m):
    """
    Return fluorescence intensity F(t) = m*[S1S2F] + F0
    """
    # Initial conditions [L, R, LR, LRin, S1, S1S2F]
    y0 = [L0, R0, 0.0, 0.0, S1_0, 0.0]
    
    # Solve ODE
    t_span = np.linspace(0, max(t_sec), 1000)
    sol = odeint(model, y0, t_span, args=(k1, k1r, k2, k2r, k3), rtol=1e-6, atol=1e-9)
    
    # Extract S1S2F concentration
    S1S2F_curve = sol[:, 5]
    
    # Interpolate to experimental time points
    S1S2F_expt = np.interp(t_sec, t_span, S1S2F_curve)
    
    # Calculate fluorescence intensity = m*[S1S2F] + background fluorescence(30)
    F = m * S1S2F_expt + 30
    return F

# Parameter initial guesses
k1_guess = 1e6       # M⁻¹s⁻¹ (association rate)
k1r_guess = 0.01     # s⁻¹ (dissociation rate)
k2_guess = 0.001     # s⁻¹ (forward flip rate)
k2r_guess = 0.0001   # s⁻¹ (reverse flip rate)
k3_guess = 1e6       # M⁻¹s⁻¹ (catalytic reaction)
m_guess = 1e9        # Fluorescence scaling factor

# Fit curve
params_opt, params_cov = curve_fit(
    fluorescence_curve,
    experimental_time_sec,
    experimental_fluorescence,
    p0=[k1_guess, k1r_guess, k2_guess, k2r_guess, k3_guess, m_guess],
    bounds=(
        [1e4, 1e-6, 1e-7, 1e-7, 1e4, 1e8],   # Lower bounds
        [1e7, 1e-1, 1e-3, 1e-3, 1e7, 1e13]    # Upper bounds
    ),
    maxfev=100000
)

# Extract fitted parameters
k1_fit, k1r_fit, k2_fit, k2r_fit, k3_fit, m_fit = params_opt
print(f"Fitted parameters:")
print(f"k1   = {k1_fit:.3e} M⁻¹s⁻¹")
print(f"k1r  = {k1r_fit:.3e} s⁻¹")
print(f"k2   = {k2_fit:.3e} s⁻¹")
print(f"k2r  = {k2r_fit:.3e} s⁻¹")
print(f"k3   = {k3_fit:.3e} M⁻¹s⁻¹")
print(f"m    = {m_fit:.3e} (fluorescence intensity/M)")

# Calculate fitted values
fitted_fluorescence = fluorescence_curve(
    experimental_time_sec, 
    k1_fit, k1r_fit, k2_fit, k2r_fit, k3_fit, m_fit
)

# Calculate R²
ss_res = np.sum((experimental_fluorescence - fitted_fluorescence)**2)
ss_tot = np.sum((experimental_fluorescence - np.mean(experimental_fluorescence))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\nGoodness of fit:")
print(f"R² = {r_squared:.6f}")

# Visualize results
plt.figure(figsize=(12, 8))

# Experimental data
plt.scatter(experimental_time_min, experimental_fluorescence, 
            s=80, c='red', edgecolor='k', label='Experimental data', zorder=10)

# Plot fitted curve
t_dense_min = np.linspace(0, 360, 1000)
t_dense_sec = t_dense_min * 60
fitted_dense = fluorescence_curve(
    t_dense_sec, k1_fit, k1r_fit, k2_fit, k2r_fit, k3_fit, m_fit
)
plt.plot(t_dense_min, fitted_dense, 'b-', linewidth=2.5, label='Model fit')

# Add model equations and parameter information
param_text = (
    f'$R^2 = {r_squared:.5f}$\n\n'
    f'Model parameters:\n'
    f'$k_1 = {k1_fit:.2e}\ M^{{-1}}s^{{-1}}$\n'
    f'$k_{{ -1}} = {k1r_fit:.2e}\ s^{{-1}}$\n'
    f'$k_2 = {k2_fit:.2e}\ s^{{-1}}$\n'
    f'$k_{{ -2}} = {k2r_fit:.2e}\ s^{{-1}}$\n'
    f'$k_3 = {k3_fit:.2e}\ M^{{-1}}s^{{-1}}$\n'
    f'$m = {m_fit:.2e}$'
)
plt.annotate(param_text, xy=(0.68, 0.25), xycoords='axes fraction', 
             bbox=dict(boxstyle='round', alpha=0.2, facecolor='white'))

plt.title('Fluorescence Intensity Kinetics Curve Fitting', fontsize=14)
plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Fluorescence Intensity', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.2)
plt.tight_layout()

# Create data array to save (time in minutes and fitted fluorescence)
fit_curve_data = np.column_stack((t_dense_min, fitted_dense))

# Save to file
np.savetxt('fitted_curve_data.txt', fit_curve_data, 
           header='Time(min)\tFluorescence', 
           fmt='%.6f', 
           delimiter='\t',
           comments='')

print("\nFitted curve data saved to: fitted_curve_data.txt")
plt.savefig('fluorescence_fit.png', dpi=300)
plt.show()
