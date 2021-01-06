% Repeated optimization runs, store current maximum
% --------------------------------------------------------------
% Refocusing:
% params = [tref, pref, aref, tfp, tacq, Rs(Qsw_on, Qsw_off, Tx_on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------
% Excitation:
% params = [texc, pexc, tacq, Rs(Qsw_on, Qsw_off, Tx_on), neff] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function opt_exc_pulse_tuned_inv_repeat(filname,pulse_nums)

close all;
tmp=load(filname); results_curr=tmp.results;

num_init=1e2;

results={};
for j=pulse_nums
    
    % Load existing pulse parameters
    params=results_curr{j,8}; nexc=length(params.texc);
    sp=results_curr{j,9}; pp=results_curr{j,10};
    sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; % Turn off plots
    
    % Calculate existing refocusing axis
    [neff,~]=calc_rot_axis_tuned_probe_lp(params,sp,pp);
    params.neff=neff; params.pcycle=0;
    
    % Calculate existing transverse magnetization
    [mrx,SNR,~]=calc_masy_tuned_probe_lp(params,sp,pp);
    params.mrx=mrx; params.SNR=SNR;
    
    % Run optimization
    %params.pexc=0.9*(2*pi-params.pexc)+0.1*2*pi*rand(1,nexc); % Start close to PI phase
    %params.pexc=2*pi*rand(1,nexc); % Start from random initial conditions
    %tmp=load('results_circsim_1_10_run2_reopt_inv2.mat'); tmp=tmp.results;
    %params.pexc=0.9*tmp{3,2}+0.1*2*pi*rand(1,nexc);
    
    pexc_orig=params.pexc; val_min=0;
    for i=1:num_init % Look for good initial conditions
        if i==1
            %Close to PI
            %params.pexc=0.9*(2*pi-pexc_orig)+0.1*2*pi*rand(1,nexc);
            %Close to PAP
            params.pexc=0.7*mod(pi+pexc_orig,2*pi)+0.3*2*pi*rand(1,nexc);
            %Random
            %params.pexc=2*pi*rand(1,nexc);
        else
            %params.pexc=2*pi*rand(1,nexc);
            params.pexc=0.7*pexc_best+0.3*2*pi*rand(1,nexc); % Start from random initial conditions
        end
        [mrx,SNR,~]=calc_masy_tuned_probe_lp(params,sp,pp);
        val=trapz(sp.del_w,abs(params.mrx+mrx))+0.8*abs(SNR/1e8-params.SNR);
        if i==1 || val<val_min
            val_min=val; disp(val_min)
            pexc_best=params.pexc;
        end
%         disp(i)
    end
    params.pexc=pexc_best;
    
    [out]=opt_exc_pulse_circsim_inv_lp(params,sp,pp);
    
    results{j,1}=out.texc;
    results{j,2}=out.pexc;
    results{j,3}=out.aexc;
    results{j,4}=params.tref;
    results{j,5}=params.pref;
    results{j,6}=params.aref;
    results{j,7}=out.axis_rms;
	resulyd{j,8}=out.params;
    results{j,9}=out.sp;
    results{j,10}=out.pp;
    save([filname '_inv1.mat'], 'results');
    
    disp(out.axis_rms)
    disp(j)
end