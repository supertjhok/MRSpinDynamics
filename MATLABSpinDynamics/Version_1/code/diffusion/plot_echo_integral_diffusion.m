% Post-process data from compare_diffusion function
% Plot echo integral as a function of eta

function plot_echo_integral_diffusion(expt_num,echo_num,clr)

tmp=load('compare_diffusion_results.mat');
diff_results=tmp.diff_results;

eta=linspace(0,3,1e2);
echo_int=zeros(1,length(eta));

tmp=diff_results{expt_num,2}; % disttimes
a_l=tmp(echo_num,:);
eta_l=1./diff_results{expt_num,1}; % axistime

for j=1:length(eta)
    echo_int(j)=sum(a_l.*exp(-eta_l*eta(j)));
end

figure(1);
%semilogy(eta,echo_int,clr); hold on; % Non-normalized
semilogy(eta,echo_int/echo_int(1),clr); hold on; % Normalized
%semilogy(eta,echo_int(1)*exp(-eta),'k--'); % On-resonance case
xlabel('\eta = \gamma^{2}g^{2}Dt_{E}^{3}N / 12');
ylabel('Echo amplitude')