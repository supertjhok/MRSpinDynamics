% Calculate diffusion spectra for some example sequences
% Soumyajit Mandal, 03/04/11
function compare_diffusion(expt,Nfinal)

close all;
% Common parameters
T_90=pi/2; % length of nominal 90 degree pulse (0 dB), sec
T_E=14*T_90; % echo separation

% Define pulses
% -----------------------------------------------
% RP2-1.0
tref_1=2*T_90*[0.14 0.72 0.14];
pref_1=(pi/2)*[3,1,3];
aref_1=ones(1,3);

% CP-M15
tmp=load('..\dat_files\exc_timings.mat');
results=tmp.results;

texc_15=T_90*results{16,2};
pexc_15=results{16,3};
aexc_15=ones(1,20);

% VAN_EXC
tmp=load('..\dat_files\van_exc.mat');
van_exc=tmp.van_exc;

texc_van=0.2*T_90*ones(1,1e2);
pexc_van=van_exc(:,2)'*(pi/180); 
aexc_van=van_exc(:,1)';

% PULSE-4
% tmp=load('..\dat_files\longerpulses.mat');
% pulse_4=tmp.pulse_4;
% comp_4=pulse_4(:,1)+1i*pulse_4(:,2);
% comp_4=comp_4';
% 
% tref_4=T_90*0.2*ones(1,40);
% pref_4=phase(comp_4)+pi/2;
% aref_4=abs(comp_4)./max(abs(comp_4));
% -----------------------------------------------

switch expt
    
    case 1
        disp('Rectangular excitation pulse (0 dB), rectangular refocusing pulses (0 dB)')
        
        [axistime,disttimes]=...
            coherences_new(T_90,pi,1,2*T_90,pi/2,1,T_E,Nfinal);
        save_data(expt,axistime,disttimes);
        
    case 2
        disp('Rectangular excitation pulse (20 dB), RP2 refocusing pulses (0 dB)')
        
        [axistime,disttimes]=...
            coherences_new(T_90/10,pi,10,tref_1,pref_1,aref_1,T_E,Nfinal);
        save_data(expt,axistime,disttimes);
        
    case 3
        disp('CP-M15 excitation pulse (0 dB), RP2 refocusing pulses (0 dB)')
        
        [axistime,disttimes]=...
            coherences_new(texc_15,pexc_15,aexc_15,tref_1,pref_1,aref_1,T_E,Nfinal);
        save_data(expt,axistime,disttimes);
        
    case 4
        disp('Rectangular excitation pulse (20 dB), PULSE-4 refocusing pulses (0 dB)')
        
        [axistime,disttimes]=...
            coherences_new(T_90/10,pi,10,tref_4,pref_4,aref_4,T_E,Nfinal);
        save_data(expt,axistime,disttimes);
        
    case 5
        disp('CP-M15 excitation pulse (0 dB), PULSE-4 refocusing pulses (0 dB)')
        
        [axistime,disttimes]=...
            coherences_new(texc_15,pexc_15,aexc_15,tref_4,pref_4,aref_4,T_E,Nfinal);
        save_data(expt,axistime,disttimes);
        
    case 6
        disp('Rectangular excitation pulse (0 dB), rectangular 135 degree refocusing pulses (0 dB)')
        
        [axistime,disttimes]=...
            coherences_new(T_90,pi,1,1.5*T_90,pi/2,1,T_E,Nfinal);
        save_data(expt,axistime,disttimes);
        
    case 7
        disp('Rectangular excitation pulse (0 dB), rectangular 90 degree refocusing pulses (0 dB)')
        
        [axistime,disttimes]=...
            coherences_new(T_90,pi,1,T_90,pi/2,1,T_E,Nfinal);
        save_data(expt,axistime,disttimes);
        
    case 8
        disp('VAN_EXC excitation pulse (0 dB), RP2-1.0 refocusing pulses (0 dB)')
        
        [axistime,disttimes]=...
            coherences_new(texc_van,pexc_van,aexc_van,tref_1,pref_1-pi/2,aref_1,T_E,Nfinal);
        save_data(expt,axistime,disttimes);
end

function save_data(expt,axistime,disttimes)

tmp=load('compare_diffusion_results.mat');
diff_results=tmp.diff_results;

diff_results{expt,1}=axistime; diff_results{expt,2}=disttimes;
save('compare_diffusion_results.mat','diff_results');