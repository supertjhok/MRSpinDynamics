% Repeated optimization runs, store current maximum
% --------------------------------------------------------------
% Refocusing:
% params = [tref, pref, aref, tfp, tacq, Rs(Qsw on, Qsw off, Tx on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------
% Excitation:
% params = [texc, pexc, tacq, Rs(Qsw on, Qsw off, Tx on), neff] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function opt_exc_pulse_circsim_lp_1_repeat(filname,refpulse_num,lengthExc)

% Load the refocusing pulse
% switch results_num
%     % No symmetry constraints
%     case 1 % fscale = 1, 1:8 tap on transmitter
%         filname=['results_ref_pulse_circsim_lp_1_16_run1']; % 16 segments, length = 1*T_80
% end

tmp=load(filname); results=tmp.results;
params=results{refpulse_num,5};
params.trd = pi; % Delay to allow excitation pulse to ring down
sp=results{refpulse_num,6}; pp=results{refpulse_num,7};

% Calculate refocusing axis
[neff,~]=calc_rot_axis_tuned_probe_lp(params,sp,pp);
params.neff=neff;

% Parameters of the excitation pulse
lexc=sp.cycle*(pi/2)/pp.T_90; % Each segment is 1 RF cycle long
nexc=round(lengthExc/lexc); % Pulse is n*T_180 long 
texc=lexc*ones(1,nexc);

% phi_orig=results{refpulse_num,2}; % Original segment phases
% [phiQ_ref]=quantize_phase(phi_orig,sp,pp); %Discretize phases according to sp.N
% params.pref = phiQ_ref;

count=1;
results={};
for count = 1:1
    params.texc=texc;
    if count==1
        params.pexc=2*pi*rand(1,nexc);
    else
        %params.pexc=0.9*pexc0+0.1*2*pi*rand(1,nexc);
        params.pexc=2*pi*rand(1,nexc);
    end
    
    % Run optimization
    [out]=opt_exc_pulse_circsim_lp_1(params,sp,pp);
    
    results{count,1}=out.texc;
    results{count,2}=out.pexc;
    results{count,3}=out.aexc;
    results{count,4}=params.tref;
    results{count,5}=params.pref;
    results{count,6}=params.aref;
    results{count,7}=out.axis_rms;
    results{count,8}=out.params;
    results{count,9}=out.sp;
    results{count,10}=out.pp;
    save(['results_circsim_Sep19_1050ms_doublecycle_' num2str(lengthExc) '.mat'], 'results');
    
    if (count==1) || (out.axis_rms>axis_rms_max)
        axis_rms_max=out.axis_rms;
        pexc0=out.pexc;
    end
    
    disp(out.axis_rms)
    disp(count)
    
    % Get new initial conditions and repeat optimization
    count=count+1;
end