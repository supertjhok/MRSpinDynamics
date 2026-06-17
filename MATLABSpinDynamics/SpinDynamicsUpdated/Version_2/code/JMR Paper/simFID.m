[sp, pp] = set_params_FID

params.tp = [25e-6 (25e-6*3) 50e-6 25e-6*6];
params.phi = [0 0 pi/2 0];
params.amp = [1 0 1 0];
params.acq = [0 0 0 1];
params.grad = [0 0 0 0];
params.len_acq = (25e-6*6); 
numpts=2000; 
maxoffs=1;
params.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts);
params.del_wg = zeros(size(params.del_w));
params.w_1 = ones(size(params.del_w));
params.T1n = 200000; 
params.T2n = 200000;

params.m0 = 1; % Initial magnetization vector amplitude
params.mth = 1; % Thermal magnetization vector amplitude

[macq]=sim_spin_dynamics_arb7(params)

figure
plot((real(fft(macq))));
hold on
plot((imag(fft(macq))));
title('fft')

figure
plot(real(macq))
hold on
plot(imag(macq))
title('macq')