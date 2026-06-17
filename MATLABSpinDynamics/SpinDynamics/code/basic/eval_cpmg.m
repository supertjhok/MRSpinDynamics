% Theoretical evaluation of various excitation and refocusing pulses
% Soumyajit Mandal, 03/16/11

function [echo,tvect]=eval_cpmg(expt)

NE = 10;
T_180 = pi;
T_FP = 6*T_180;
delt = T_FP; len_acq=T_FP;

switch expt
    
    case 1
        display('Rectangular excitation (0 dB), rectangular refocusing (0 dB, 180 degrees)');
        
        texc=T_180/2; pexc=0; aexc=1;
        tref=T_180; pref=pi/2; aref=1;
        
        [refmat,del_w]=calc_refocusing_mat_arba(tref,pref,aref,NE,T_FP);
        [echo1,~,~,~]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc,aexc,refmat,del_w,delt,len_acq);
        [echo2,tvect,~,~]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc+pi,aexc,refmat,del_w,delt,len_acq);
        echo=(echo1-echo2)/2; % Phase cycle
        
        figure(1);
        plot(tvect/T_180,abs(echo)); hold on;
        xlabel('t / T_{180}');
        
    case 2
        display('VAN_EXC (0 dB), RP2-1.0 refocusing (0 dB)');
        
        tmp=load('van_exc.mat'); van_exc=tmp.van_exc;
        texc=0.1*T_180*ones(1,1e2); pexc=van_exc(:,2)'*(pi/180); aexc=van_exc(:,1)';
        tref=T_180*[0.14 0.72 0.14]; pref=(pi/2)*[2 0 2]; aref=[1 1 1];
        
        [refmat,del_w]=calc_refocusing_mat_arba(tref,pref,aref,NE,T_FP);
        [echo1,~,~,~]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc,aexc,refmat,del_w,delt,len_acq);
        [echo2,tvect,~,~]=cpmg_van_spin_dynamics_refmat_arba(texc,-pexc,aexc,refmat,del_w,delt,len_acq);
        echo=(echo1-echo2)/2; % Phase cycle
        
        figure(1);
        plot(tvect/T_180,abs(echo)); hold on;
        xlabel('t / T_{180}');
        
    case 3
        display('CP-M15 (0 dB), RP2-1.0 refocusing (0 dB)');
        
        tmp=load('exc_timings.mat'); results=tmp.results;
        texc=(T_180/2)*results{16,2}; pexc=results{16,3}; aexc=ones(1,20);
        tref=T_180*[0.14 0.72 0.14]; pref=(pi/2)*[3 1 3]; aref=[1 1 1];
        
        [refmat,del_w]=calc_refocusing_mat_arba(tref,pref,aref,NE,T_FP);
        [echo1,~,~,~]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc,aexc,refmat,del_w,delt,len_acq);
        [echo2,tvect,~,~]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc+pi,aexc,refmat,del_w,delt,len_acq);
        echo=(echo1-echo2)/2; % Phase cycle
        
        figure(1);
        plot(tvect/T_180,abs(echo)); hold on;
        xlabel('t / T_{180}');
        
    case 4
        display('Rectangular (20 dB), RP2-1.0 refocusing (0 dB)');
        
        texc=T_180/20; pexc=0; aexc=10;
        tref=T_180*[0.14 0.72 0.14]; pref=(pi/2)*[3 1 3]; aref=[1 1 1];
        
        [refmat,del_w]=calc_refocusing_mat_arba(tref,pref,aref,NE,T_FP);
        [echo1,~,~,~]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc,aexc,refmat,del_w,delt,len_acq);
        [echo2,tvect,~,~]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc+pi,aexc,refmat,del_w,delt,len_acq);
        echo=(echo1-echo2)/2; % Phase cycle
        
        figure(1);
        plot(tvect/T_180,abs(echo)); hold on;
        xlabel('t / T_{180}');
end