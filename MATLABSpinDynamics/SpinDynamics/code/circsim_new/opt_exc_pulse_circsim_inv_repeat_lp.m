% Repeated optimization runs, store current maximum
% --------------------------------------------------------------
% Refocusing:
% params = [tref, pref, aref, tfp, tacq, Rs(off,on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------
% Excitation:
% params = [texc, pexc, tacq, Rs(off,on), neff] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function opt_exc_pulse_circsim_inv_repeat_lp(filname,pulse_num)

close all;
tmp=load(filname); results_curr=tmp.results;
% params.tacq=tacq;

% techo = round(techo*(pp.T_90/(pi/2))/sp.cycle)*sp.cycle*(pi/2/pp.T_90);
% params.tfp=(techo-pi)/2;
% params.tacq=round(tacq*(pp.T_90/(pi/2))/sp.cycle)*sp.cycle*(pi/2/pp.T_90);
results={}; cnt=1;
while(1)
    
    % Load existing pulse parameters
    params.texc=results_curr{pulse_num,1}; nexc=length(params.texc);
    params.pexc=results_curr{pulse_num,2};
    params.aexc=results_curr{pulse_num,3};
    params.tref=results_curr{pulse_num,4};
    params.pref=results_curr{pulse_num,5};
    params.aref=results_curr{pulse_num,6};
    sp=results_curr{pulse_num,9}; pp=results_curr{pulse_num,10};
    sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; % Turn off plots
    params = results_curr{pulse_num,8};
    
    
    % Calculate existing refocusing axis
    [neff,~]=calc_rot_axis_tuned_probe_lp(params,sp,pp);
    params.neff=neff; params.pcycle=0;
    
    % Calculate existing transverse magnetization
    [mrx,SNR,~]=calc_masy_tuned_probe_lp(params,sp,pp);
    params.mrx=mrx; params.SNR=SNR;
    
    % Run optimization
    params.pexc=0.9*(2*pi-params.pexc)+0.1*2*pi*rand(1,nexc); % Start close to PI phase
    %params.pexc=2*pi*rand(1,nexc); % Start from random initial conditions
    
    [out]=opt_exc_pulse_circsim_inv_lp(params,sp,pp);
    
    results{cnt,1}=out.texc;
    results{cnt,2}=out.pexc;
    results{cnt,3}=out.aexc;
    results{cnt,4}=params.tref;
    results{cnt,5}=params.pref;
    results{cnt,6}=params.aref;
    results{cnt,7}=out.axis_rms;
    results{cnt,8}=out.sp;
    results{cnt,9}=out.pp;
    save([filname '_' num2str(pulse_num) '_inv1.mat'], 'results');
    
    disp(out.axis_rms)
    disp(cnt)
    
    cnt=cnt+1;
end