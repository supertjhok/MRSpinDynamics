% Repeated optimization runs, store current maximum
% --------------------------------------------------------------
% Refocusing:
% params = [tref, pref, aref, tfp, tacq, Rs(Qsw on, Qsw off, Tx on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------
% Excitation:
% params = [texc, pexc, tacq, Rs(Qsw on, Qsw off, Tx on), neff] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function opt_exc_pulse_tuned_repeat(infile,refpulse_num,lengthExc,outfile)

% Number of iteration runs (should be integer multiple of parpool size)
numiter=24;

% Load the refocusing pulse
tmp=load(infile); results_all=tmp.results;
results=results_all{refpulse_num};
params=results{5};
sp=results{6}; pp=results{7};

% Calculate refocusing axis
[neff,~]=calc_rot_axis_tuned_probe_lp_Orig(params,sp,pp);
params.neff=neff;

% Parameters of the excitation pulse
%lexc=sp.cycle*(pi/2)/pp.T_90; % Each segment is 1 RF cycle long
lexc=2*sp.cycle*(pi/2)/pp.T_90; % Each segment is 2 RF cycles long
nexc=round(lengthExc/lexc); % Pulse is n*T_180 long 
disp(['Number of optimization variables = ' num2str(nexc)]);

% Convert delays to absolute time (needed for circuit solver code)
params.texc=lexc*ones(1,nexc)*pp.T_90/(pi/2);
params.trd=(pi/2)*pp.T_90/(pi/2); % Delay to allow excitation pulse to ring down

% phi_orig=results{refpulse_num,2}; % Original segment phases
% [phiQ_ref]=quantize_phase(phi_orig,sp,pp); %Discretize phases according to sp.N
% params.pref = phiQ_ref;

results=cell(numiter,1);
parfor count = 1:numiter	
	disp(['Starting run ' num2str(count)]);
    params_curr=params;
    params_curr.pexc=2*pi*rand(1,nexc);
    
    % Run optimization
    [out]=opt_exc_pulse_tuned(params_curr,sp,pp);
    
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