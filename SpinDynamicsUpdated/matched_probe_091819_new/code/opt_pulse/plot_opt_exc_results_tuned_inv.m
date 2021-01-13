% Plot original and inverse AMEX pulses

function plot_opt_exc_results_tuned_inv(filname,pulse_num)

% Original pulse
[mrx1,masy1,echo_rx1,tvect,SNR1] = plot_opt_exc_results_tuned(filname,pulse_num);

% Inverted pulse
[mrx2,masy2,echo_rx2,~,SNR2] = plot_opt_exc_results_tuned([filname '_inv'],pulse_num);

% Correlation coefficent
rho = corr(mrx1',mrx2');

% Display results
disp(['SNR of original pulse = ' num2str(SNR1)])
disp(['SNR of inverse pulse = ' num2str(SNR2)])
disp(['Correlation coefficient = ' num2str(rho)])

