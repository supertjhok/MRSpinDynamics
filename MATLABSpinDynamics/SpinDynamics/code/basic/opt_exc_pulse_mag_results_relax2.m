% Plot OCT pulse performance including relaxation relaxation
% quant -> use phase quantization
% NE -> number of echoes

function [masy,del_w,tot_time]=...
    opt_exc_pulse_mag_results_relax2(expt,quant,pulse_num,NE,T1,T2)

T_90=pi/2;
len_acq=4*pi;

num_pexc=128; % Number of phase quantization steps
switch expt
    
    case 1
        % Default / rectangular case
        tref=pi*[3 0.25 0.25 0.25 0.25 3]; % 180 degrees
        pref=pi*[0 0 0 0 0 0];
        aref=[0 1 1 1 1 0];
        
        texc=pi*[0.25 0.25];
        pexc=pi*[1 1]/2;
        aexc=[1 1];
        
    case 2
        % Default / rectangular case
        tref=pi*[3.125 0.25 0.25 0.25 3.125]; % 135 degrees
        pref=pi*[0 0 0 0 0];
        aref=[0 1 1 1 0];
        
        texc=pi*[0.25 0.25];
        pexc=pi*[1 1]/2;
        aexc=[1 1];
        
    case 3
        % Van's OCT excitation pulse
        tref=pi*[3 0.14 0.24 0.24 0.24 0.14 3];
        pref=pi*[0 1 0 0 0 1 0];
        aref=[0 1 1 1 1 1 0];
        
        tmp=load(fullfile('dat_files','van_exc.mat'));
        van_exc=tmp.van_exc;
        
        texc=0.2*T_90*ones(1,1e2);
        pexc=van_exc(:,2)'*(pi/180);
        aexc=van_exc(:,1)';
        %aexc=ones(1,1e2); % No amplitude modulation
        
    case 4
        % Original constant-amplitude OCT excitation pulse
        tref=pi*[3 0.14 0.24 0.24 0.24 0.14 3];
        pref=pi*[0 1 0 0 0 1 0];
        aref=[0 1 1 1 1 1 0];
        
        tmp=load(fullfile('dat_files','results_mag3_rms_2.mat'));
        texc=tmp.texc;
        pexc=tmp.pexc;
        aexc=ones(1,length(texc));
        
    case 5
        % New constant-amplitude OCT excitation pulses
        tref=pi*[3 0.14 0.24 0.24 0.24 0.14 3];
        pref=pi*[0 1 0 0 0 1 0];
        aref=[0 1 1 1 1 1 0];
        
        tmp=load(fullfile('dat_files','results_mag_all.mat'));
        results=tmp.results_sort;
        texc=results{pulse_num,1};
        pexc=results{pulse_num,2};
        aexc=ones(1,length(texc));
        
    case 6
        % Constant-amplitude OCT excitation pulses optimized for relaxation
        tref=pi*[3 0.14 0.24 0.24 0.24 0.14 3];
        pref=pi*[0 1 0 0 0 1 0];
        aref=[0 1 1 1 1 1 0];
        
        tmp=load(fullfile('dat_files','results_mag_relax.mat'));
        results=tmp.results;
        texc=results{pulse_num,1};
        pexc=results{pulse_num,2};
        aexc=ones(1,length(texc));
        
end

if quant
    del=2*pi/num_pexc;
    pexc=del*round(pexc/del); % Phase quantization
end

oexc=zeros(1,length(texc)); oref=zeros(1,length(tref));
tot_time=sum(texc)+NE*sum(tref); % Total sequence duration

[macq1,del_w]=...
    cpmg_van_spin_dynamics_arb(texc,pexc,aexc,oexc,tref,pref,aref,oref,NE,len_acq,T1,T2);

if length(texc)>2  % OCT excitation pulse, use phase inversion instead of cycling
    [macq2,~]=...
        cpmg_van_spin_dynamics_arb(texc,-pexc,aexc,oexc,tref,pref,aref,oref,NE,len_acq,T1,T2);
else
    [macq2,~]=...
        cpmg_van_spin_dynamics_arb(texc,pexc+pi,aexc,oexc,tref,pref,aref,oref,NE,len_acq,T1,T2);
end

masy1=macq1; masy2=macq2; % Asymptotic echo
masy=(masy1-masy2)/2; % Phase cycling / inversion