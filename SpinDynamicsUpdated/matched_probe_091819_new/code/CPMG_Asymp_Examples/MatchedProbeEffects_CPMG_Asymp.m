close all;

[sp, pp] = set_params_matched_Orig; % Define system parameters
[mrx,masy,SNR]=calc_masy_matched_probe_Orig(sp,pp); % Simulate narrowband system

figure;
plot(sp.del_w,real(masy),'LineWidth',1); 
hold on; 
plot(sp.del_w,imag(masy),'LineWidth',1);

% Divide received magnetization by peak TF gain (=0.5*sqrt(Rs/Rc)) for clarity
Grx_max=0.5*sqrt(sp.Rs/sp.R);
plot(sp.del_w,real(mrx)/Grx_max,'LineWidth',1);
plot(sp.del_w,imag(mrx)/Grx_max,'LineWidth',1);

title('Asymptotic magnetization')
xlabel('\Delta\omega_{0}/\omega_{1,max}')
ylabel('M_{asy}, M_{rx}')
legend({'Real(M_{asy})','Imag(M_{asy})','Real(M_{rx})','Imag(M_{rx})'})
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
% export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\echoMagAsympMatched.pdf

% Calculate time-domain echo
[echo_rx,tvect]=calc_time_domain_echo(mrx/Grx_max,sp.del_w,1,1);