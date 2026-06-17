function plot_diffusion_theo_expt(expt_nums,echo_num,T2)

%close all;
[eta1,eint1]=plot_diffusion(expt_nums(1),T2); % Rectangular, 180
[eta2,eint2]=plot_diffusion(expt_nums(2),T2); % RP2-1.0
[eta3,eint3]=plot_diffusion(expt_nums(3),T2); % Rectangular, 135

% Theory
if echo_num < 13
    plot_echo_integral_diffusion(1,echo_num,'b--'); % Rectangular, 180
    plot_echo_integral_diffusion(2,echo_num,'r--'); % RP2-1.0
    plot_echo_integral_diffusion(6,echo_num,'k--'); % Rectangular, 135
end

% Experiment
semilogy(eta1(echo_num,:),eint1(echo_num,:)/eint1(echo_num,1),'bo')
semilogy(eta2(echo_num,:),eint2(echo_num,:)/eint2(echo_num,1),'r*')
semilogy(eta3(echo_num,:),eint3(echo_num,:)/eint3(echo_num,1),'kd')

xlim([0,ceil(max(eta1(echo_num,:)))])
title(['N = ' num2str(echo_num)])