%% Track MR - Vertical Drone Motion with Thrust-Actuator Lag
% State model:   z_dot = v
%                m*v_dot = T - m*g - c*v*|v|
%                tau_T*T_dot = Tcmd(t) - T
% MMM 5162 - Modelling and Simulation | Mini Project 1
% Student: Omar Yahya Hadi, ID 481011766 (S = 11)

clear; clc; close all;

%% ---- Student-specific parameters (S = 11) ----
S   = 11;
g   = 9.81;                 % m/s^2
m   = 1.60 + 0.03*S;        % kg
c   = 0.120 + 0.005*S;      % quadratic drag coefficient
tau = 0.160 + 0.005*S;      % s, thrust time constant
r   = 1.180 + 0.005*S;      % pulse thrust ratio
pulse_dur = 1.40 + 0.03*S;  % s

t1   = 1.0;                 % pulse start (s)
t2   = t1 + pulse_dur;      % pulse end   (s)
tEnd = 8.0;                 % simulation horizon (s)

% Automatic code check - reject non-physical inputs
assert(m > 0 && tau > 0, 'Mass and thrust time constant must be positive.');

fprintf('m=%.4f kg, c=%.4f, tau=%.4f s, r=%.4f, pulse=[%.2f,%.2f] s\n', ...
        m, c, tau, r, t1, t2);

x0 = [0; 0; m*g];           % z(0)=0, v(0)=0, T(0)=mg (hover)

%% ---- Standard solver: ode45 (reference solution) ----
opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-12, 'MaxStep', 0.01);
[t_ref, x_ref] = ode45(@(t, x) drone_rhs(t, x, m, g, c, tau, r, t1, t2), ...
                        [0 tEnd], x0, opts);

z_ref = x_ref(:,1);  v_ref = x_ref(:,2);  T_ref = x_ref(:,3);
[vmax_ref, idx] = max(v_ref);
tvmax_ref = t_ref(idx);
z8_ref = z_ref(end);

fprintf('[ode45 reference] max v = %.4f m/s at t = %.3f s ; z(8s) = %.4f m\n', ...
        vmax_ref, tvmax_ref, z8_ref);

%% ---- Forward Euler (fixed step) ----
dts = [0.05 0.01 0.001];
euler = struct();
for k = 1:numel(dts)
    dt = dts(k);
    [te, ze, ve, Te] = euler_solve(dt, tEnd, x0, m, g, c, tau, r, t1, t2);
    [vmax, i2] = max(ve);
    tvmax = te(i2);
    z8 = ze(end);
    err_v = 100*(vmax - vmax_ref)/vmax_ref;
    err_z = 100*(z8 - z8_ref)/z8_ref;
    fprintf('[Euler dt=%.3f] max v = %.4f m/s at t = %.3f s ; z(8s) = %.4f m ; err_v=%.2f%% err_z=%.2f%%\n', ...
            dt, vmax, tvmax, z8, err_v, err_z);
    euler(k).dt = dt; euler(k).t = te; euler(k).z = ze; euler(k).v = ve; euler(k).T = Te;
end

%% ---- Plots: ode45 vs forward Euler ----
figure('Position',[100 100 650 650]);
subplot(3,1,1); hold on; grid on;
plot(t_ref, z_ref, 'k-', 'LineWidth', 1.8, 'DisplayName', 'ode45 (reference)');
for k = 1:numel(dts)-1   % plot dt=0.05, 0.01 (dt=0.001 visually coincides with reference)
    plot(euler(k).t, euler(k).z, '--', 'LineWidth', 1.1, ...
        'DisplayName', sprintf('Euler dt=%.2fs', euler(k).dt));
end
ylabel('Altitude z (m)'); legend('Location','southeast');
title('ode45 (Standard Solver) vs Forward Euler - State Trajectories');

subplot(3,1,2); hold on; grid on;
plot(t_ref, v_ref, 'k-', 'LineWidth', 1.8);
for k = 1:numel(dts)-1
    plot(euler(k).t, euler(k).v, '--', 'LineWidth', 1.1);
end
ylabel('Velocity v (m/s)');

subplot(3,1,3); hold on; grid on;
plot(t_ref, T_ref, 'k-', 'LineWidth', 1.8);
for k = 1:numel(dts)-1
    plot(euler(k).t, euler(k).T, '--', 'LineWidth', 1.1);
end
ylabel('Thrust T (N)'); xlabel('Time (s)');

saveas(gcf, 'fig1_states.png');

%% ---- Figure 2: Standard solver vs forward Euler - velocity zoom ----
figure('Position',[100 100 650 350]); hold on; grid on;
plot(t_ref, v_ref, 'k-', 'LineWidth', 1.8, 'DisplayName', 'ode45 (reference)');
for k = 1:numel(dts)
    plot(euler(k).t, euler(k).v, '--', 'LineWidth', 1.1, ...
        'DisplayName', sprintf('Euler dt=%.3fs', euler(k).dt));
end
xlim([0 4]);
xlabel('Time (s)'); ylabel('Velocity v (m/s)');
title('Standard Solver (ode45) vs Forward Euler - Climb Velocity');
legend('Location','southeast');

saveas(gcf, 'fig2_velocity_zoom.png');

%% ---- Local functions ----
function dxdt = drone_rhs(t, x, m, g, c, tau, r, t1, t2)
    v = x(2); T = x(3);
    Tc = Tcmd_func(t, m, g, r, t1, t2);
    dz = v;
    dv = (T - m*g - c*v*abs(v)) / m;
    dT = (Tc - T) / tau;
    dxdt = [dz; dv; dT];
end

function Tc = Tcmd_func(t, m, g, r, t1, t2)
    if t >= t1 && t < t2
        Tc = r*m*g;
    else
        Tc = m*g;
    end
end

function [t, z, v, T] = euler_solve(dt, tEnd, x0, m, g, c, tau, r, t1, t2)
    n = round(tEnd/dt);
    t = (0:n)' * dt;
    x = zeros(n+1, 3);
    x(1,:) = x0';
    for k = 1:n
        dxdt = drone_rhs(t(k), x(k,:)', m, g, c, tau, r, t1, t2);
        x(k+1,:) = x(k,:) + dt * dxdt';
    end
    z = x(:,1); v = x(:,2); T = x(:,3);
end
