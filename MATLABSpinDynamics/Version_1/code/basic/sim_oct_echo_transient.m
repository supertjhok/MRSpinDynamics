% Study the transient behavior of OCT excitation pulses with RP2-1.0
% refocusing pulses
% Soumyajit Mandal, 02/20/13

function sim_oct_echo_transient(pulse_num)

close all;
maxecho=4;

tmp=load('dat_files\results_mag_all.mat'); 
results_sort=tmp.results_sort;

% OCT excitation pulse
texc=results_sort{pulse_num,1}; pexc=results_sort{pulse_num,2};
aexc=ones(1,length(texc)); oexc=zeros(1,length(texc));

% RP2-1.0 refocusing pulse
tref=pi*[3 0.14 0.72 0.14 3]; pref=pi*[0 1 0 1 0];
aref=[0 1 1 1 0]; oref=[0 0 0 0 0];

% Acquisition window length, T1, T2
len_acq=4*pi; T1=1e8; T2=1e8;

for i=1:maxecho
    % Phase inversion cycle
    [macq1,del_w0]=cpmg_van_spin_dynamics_arb(texc,pexc,aexc,oexc,tref,pref,aref,oref,i,len_acq,T1,T2);
    [macq2,~]=cpmg_van_spin_dynamics_arb(texc,-pexc,aexc,oexc,tref,pref,aref,oref,i,len_acq,T1,T2);
    macq=(macq1-macq2)/2;
    
    subplot(ceil(sqrt(maxecho)),ceil(sqrt(maxecho)),i);
    plot(del_w0,real(macq),'b-'); hold on; plot(del_w0,imag(macq),'r-');
    title(['N = ' num2str(i)]);
    ylim([-0.2 1]);
end