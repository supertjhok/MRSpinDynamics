close all

[sp, pp] = set_params_matched; % Define system parameters
 
Qvec = linspace(11,200,100); % Vary coil Q
 
SNR = zeros(1, length(Qvec)); % Storage for output variables
% echo_rx = zeros(4*length(sp.del_w),length(Qvec));
% mrx = zeros(length(Qvec),length(sp.del_w));
   sp.plt_mn=0; sp.plt_tx=0; sp.plt_rx=0; sp.plt_axis=0; sp.plt_echo=0; 
    
% Run simulations
for i=1:length(Qvec)
    % Turn plotting off to reduce the number of plots
  sp.Q = Qvec(i); % Change coil Q
    
  
[mz(i,:),tvect]=calc_masy_matched_nut(sp,pp); % Simulate narrowband system

end
    
figure;
imagesc(sp.del_w,Qvec,1-abs(mz)); % Asymptotic magnetization
xlim([-2.0 2.0])
caxis([0.6 1])
ylim([min(Qvec) max(Qvec)])
xlabel('\omega / \omega_1');
ylabel('Q');
whiteBg
setSize
font
% colormap(jet(256))
h = colorbar;
set(get(h,'title'),'string','Z Magnetization');
export_fig('F:\Dropbox\Apps\Overleaf\Portable and Autonomous Magnetic Resonance\Figures\MrZMatched.pdf')


% plot(del_w_real/10^6,mz);
