close all
[sp, pp] = set_params_matched_SS; % Define system parameters
[mz,tvect]=calc_masy_matched_nut(sp,pp); % Simulate narrowband system

nutFreq = 1/(pp.T_90*4);
% *19230.77
figure;

del_w_real = (sp.del_w/(2*pi)*nutFreq);

plot(del_w_real/10^6,mz);

