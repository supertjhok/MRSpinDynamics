% Calculate asymptotic magnetization and echo for a CPMG sequence assuming
% an untuned probe
% -----------------------------------------------------------------------
%close all

[params,sp,pp] = set_params_untuned_Orig; % Define system parameters
[mrx,masy,SNR] = calc_masy_untuned_probe_lp(params,sp,pp);

SNR

figure;
plot(sp.del_w,real(masy),'LineWidth',1);
hold on;
plot(sp.del_w,imag(masy),'LineWidth',1);

% Divide received magnetization by peak TF gain (= transformer gain) for clarity
Grx_max=sp.Nrx*(sp.L1/(sp.L1+sp.L));
plot(sp.del_w,real(mrx)/Grx_max,'LineWidth',1);
plot(sp.del_w,imag(mrx)/Grx_max,'LineWidth',1);

title('Asymptotic magnetization')
xlabel('\Delta\omega_{0}/\omega_{1,max}')
ylabel('M_{asy}, M_{rx}')
legend({'Real(M_{asy})','Imag(M_{asy})','Real(M_{rx})','Imag(M_{rx})'})
set(gca,'FontSize',15); set(gca,'FontWeight','bold');

%figure; 
%plot(sp.del_w,abs(mrx))
%title('abs(M_{rx})');

% Plot time-domain echo
[echo_rx,tvect]=calc_time_domain_echo(mrx/Grx_max,sp.del_w,1,1);