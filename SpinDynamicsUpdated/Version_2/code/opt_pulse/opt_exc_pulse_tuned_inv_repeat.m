% Repeated optimization runs, store current maximum
% --------------------------------------------------------------
% Refocusing:
% params = [tref, pref, aref, tfp, tacq, Rs(Qsw_on, Qsw_off, Tx_on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------
% Excitation:
% params = [texc, pexc, tacq, Rs(Qsw_on, Qsw_off, Tx_on), neff] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function opt_exc_pulse_tuned_inv_repeat(infile,pulse_nums,outfile)

close all;
tmp=load(infile); results_orig=tmp.results;
siz=size(results_orig);

num_init=1e2;

results=cell(siz(1),siz(2));
parfor count=pulse_nums
    disp(['Starting run ' num2str(count)]);
    results_curr=results_orig{count};
    
    % Load existing pulse parameters
    params=results_curr{8}; nexc=length(params.texc);
    sp=results_curr{9}; pp=results_curr{10};
    sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; % Turn off plots
    
    % Calculate existing refocusing axis
    [neff,~]=calc_rot_axis_tuned_probe_lp_Orig(params,sp,pp);
    params.neff=neff; params.pcycle=0;
    
    % Calculate existing transverse magnetization
    [mrx,~,SNR]=calc_masy_tuned_probe_lp_Orig(params,sp,pp);
    params.mrx=mrx; params.SNR=SNR;
    
    % Run optimization
    %params.pexc=0.9*(2*pi-params.pexc)+0.1*2*pi*rand(1,nexc); % Start close to PI phase
    %params.pexc=2*pi*rand(1,nexc); % Start from random initial conditions
    
    pexc_orig=params.pexc; val_min=0;
    pexc_best=zeros(1,length(pexc_orig));
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
        [mrx,~,SNR]=calc_masy_tuned_probe_lp_Orig(params,sp,pp);
        val=trapz(sp.del_w,abs(params.mrx+mrx))+0.8*abs(SNR/1e8-params.SNR);
        if i==1 || val<val_min
            val_min=val; % disp(val_min)
            pexc_best=params.pexc;
        end
%         disp(i)
    end
    params.pexc=pexc_best;
    
    [out]=opt_exc_pulse_tuned_inv(params,sp,pp);
    
    % Create results structure
    results_curr=cell(1,10);
    results_curr{1}=out.texc;
    results_curr{2}=out.pexc;
    results_curr{3}=out.aexc;
    results_curr{4}=params.tref;
    results_curr{5}=params.pref;
    results_curr{6}=params.aref;
    results_curr{7}=out.axis_rms;
	results_curr{8}=out.params;
    results_curr{9}=out.sp;
    results_curr{10}=out.pp;

    % Store current set of results (for safety)
    outfile_curr=[outfile '_run_' num2str(count)];
    parsave(outfile_curr,results_curr);
    
    % Update overall results structure
    results{count}=results_curr;
    
    disp(out.axis_rms)
end

% Save optimization results
save(outfile,'results');