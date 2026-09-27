"""
Track MR - Vertical Drone Motion with Thrust-Actuator Lag
MMM 5162 - Modelling and Simulation | Mini Project 1
Student: Omar Yahya Hadi, ID 481011766 (S = 11)

State model:
    z_dot   = v
    m*v_dot = T - m*g - c*v*|v|
    tau*T_dot = Tcmd(t) - T

Standard solver: scipy.integrate.solve_ivp (RK45 = MATLAB's ode45 equivalent)
Also implements forward Euler at dt = 0.05, 0.01, 0.001 s for comparison.
Run this single file to reproduce Figures 1-2, the console printout,
and verification_table.txt.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Student-specific parameters (S = 11) ----------------
S = 11
g = 9.81
m   = 1.60 + 0.03*S      # kg
c   = 0.120 + 0.005*S    # quadratic drag coeff
tau = 0.160 + 0.005*S    # s, thrust time constant
r   = 1.180 + 0.005*S    # pulse thrust ratio
pulse_dur = 1.40 + 0.03*S  # s
t_pulse_start = 1.0
t_pulse_end = t_pulse_start + pulse_dur
t_end = 8.0

print(f"m={m:.4f} kg, c={c:.4f}, tau={tau:.4f} s, r={r:.4f}, pulse_dur={pulse_dur:.4f} s, "
      f"pulse window=[{t_pulse_start:.2f},{t_pulse_end:.2f}] s")

def Tcmd(t):
    if t_pulse_start <= t < t_pulse_end:
        return r*m*g
    return m*g

def rhs(t, x):
    z, v, T = x
    zdot = v
    vdot = (T - m*g - c*v*abs(v)) / m
    Tdot = (Tcmd(t) - T) / tau
    return [zdot, vdot, Tdot]

x0 = [0.0, 0.0, m*g]

# ---------------- Standard solver (RK45, tight tolerances) ----------------
sol = solve_ivp(rhs, [0, t_end], x0, method="RK45", max_step=0.01,
                 rtol=1e-9, atol=1e-12, dense_output=True)
t_ref = np.linspace(0, t_end, 4001)
z_ref, v_ref, T_ref = sol.sol(t_ref)

vmax_ref = v_ref.max()
tvmax_ref = t_ref[np.argmax(v_ref)]
z8_ref = z_ref[-1]
print(f"[RK45 reference] max v = {vmax_ref:.4f} m/s at t = {tvmax_ref:.3f} s; z(8) = {z8_ref:.4f} m")

# ---------------- Forward Euler ----------------
def euler(dt):
    n = int(round(t_end/dt))
    t = np.linspace(0, n*dt, n+1)
    x = np.zeros((n+1, 3))
    x[0] = x0
    for k in range(n):
        x[k+1] = x[k] + dt*np.array(rhs(t[k], x[k]))
    return t, x[:,0], x[:,1], x[:,2]

dts = [0.05, 0.01, 0.001]
euler_results = {}
for dt in dts:
    t_e, z_e, v_e, T_e = euler(dt)
    vmax = v_e.max()
    tvmax = t_e[np.argmax(v_e)]
    z8 = z_e[-1]
    euler_results[dt] = (t_e, z_e, v_e, T_e, vmax, tvmax, z8)
    print(f"[Euler dt={dt:>5}] max v = {vmax:.4f} m/s at t = {tvmax:.3f} s; z(8) = {z8:.4f} m; "
          f"err_vmax={100*(vmax-vmax_ref)/vmax_ref:.2f}%  err_z8={100*(z8-z8_ref)/z8_ref:.2f}%")

# ---------------- Plots ----------------
fig, axs = plt.subplots(3, 1, figsize=(6.2, 6.2), sharex=True)
axs[0].plot(t_ref, z_ref, 'k-', lw=1.8, label='ode45 (reference)')
axs[1].plot(t_ref, v_ref, 'k-', lw=1.8, label='ode45 (reference)')
axs[2].plot(t_ref, T_ref, 'k-', lw=1.8, label='ode45 (reference)')

colors = {0.05:'tab:red', 0.01:'tab:orange', 0.001:'tab:blue'}
plot_dts = [0.05, 0.01]  # dt=0.001 visually coincides with ode45; reported in table only
for dt in plot_dts:
    t_e, z_e, v_e, T_e, *_ = euler_results[dt]
    axs[0].plot(t_e, z_e, '--', color=colors[dt], lw=1.1, label=f'Euler dt={dt}s')
    axs[1].plot(t_e, v_e, '--', color=colors[dt], lw=1.1, label=f'Euler dt={dt}s')
    axs[2].plot(t_e, T_e, '--', color=colors[dt], lw=1.1, label=f'Euler dt={dt}s')

axs[0].set_ylabel('Altitude z (m)')
axs[1].set_ylabel('Velocity v (m/s)')
axs[2].set_ylabel('Thrust T (N)')
axs[2].set_xlabel('Time (s)')
for a in axs:
    a.grid(alpha=0.3)
axs[0].legend(loc='lower right', fontsize=8)
axs[0].set_title('ode45 (Standard Solver) vs Forward Euler — State Trajectories')
plt.tight_layout()
plt.savefig('fig1_states.png', dpi=150)
plt.close()

# Zoomed velocity comparison (near thrust pulse) for clarity
fig2, ax = plt.subplots(figsize=(6.2, 3.2))
ax.plot(t_ref, v_ref, 'k-', lw=1.8, label='ode45 (reference)')
for dt in dts:
    t_e, z_e, v_e, T_e, *_ = euler_results[dt]
    ax.plot(t_e, v_e, '--', color=colors[dt], lw=1.1, label=f'Euler dt={dt}s')
ax.set_xlim(0, 4)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity v (m/s)')
ax.set_title('Climb Velocity: Solver vs Euler Step Size')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('fig2_velocity_zoom.png', dpi=150)
plt.close()

# Save verification table to text for reporting
with open('verification_table.txt', 'w') as f:
    f.write("Method,dt(s),max|v|(m/s),t_at_vmax(s),z(8s)(m),err_vmax(%),err_z8(%)\n")
    f.write(f"ode45(ref),~0.01(adaptive<=),{vmax_ref:.4f},{tvmax_ref:.3f},{z8_ref:.4f},0.00,0.00\n")
    for dt in dts:
        t_e, z_e, v_e, T_e, vmax, tvmax, z8 = euler_results[dt]
        ev = 100*(vmax-vmax_ref)/vmax_ref
        ez = 100*(z8-z8_ref)/z8_ref
        f.write(f"Euler,{dt},{vmax:.4f},{tvmax:.3f},{z8:.4f},{ev:.2f},{ez:.2f}\n")

print("Done.")
