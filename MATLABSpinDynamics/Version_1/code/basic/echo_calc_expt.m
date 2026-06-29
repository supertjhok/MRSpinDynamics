function [T2,snr]=echo_calc_expt(expt_nums,necho,nignore)

nexpt=length(expt_nums);
T2=zeros(nexpt,1); snr=T2;

for i=1:nexpt
    [T2(i),snr(i)]=plot_echoes_bruker('longer_refocusing_pulses\cpmg_oneshot_sp',expt_nums(i),necho,nignore);
end