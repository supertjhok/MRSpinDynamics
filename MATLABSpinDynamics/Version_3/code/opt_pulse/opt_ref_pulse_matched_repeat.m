% Repeated optimization runs, store current maximum
% lp -> linearly polarized transmitter

function opt_ref_pulse_matched_repeat(ref_len,techo,file)

% Number of iteration runs (should be integer multiple of parpool size)
numiter=24;

[sp,pp]=set_params_matched_OCT;
sp.cycle=2*pi/sp.w0; % Length of one RF cycle
techo = round(techo*(pp.T_90/(pi/2))/sp.cycle)*sp.cycle*((pi/2)/pp.T_90);

% Design matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Find receiver transfer functions
[tf1,tf2] = matched_receiver_tf(sp,pp);
sp.tf1=tf1; sp.tf2=tf2;

% Parameters of the refocusing pulse
% ref_len=24/16;
%lref=sp.cycle*(pi/2)/pp.T_90; % Each segment is 1 RF cycle long
lref=2*sp.cycle*(pi/2)/pp.T_90; % Each segment is 2 RF cycles long
nref=round(ref_len*pi/lref); % Pulse is ref_len*T_180 long
disp(['Number of optimization variables = ' num2str(nref)]);

% Convert delays to absolute time (needed for circuit solver code)
pp.tref=lref*ones(1,nref)*pp.T_90/(pi/2); % Pulse
pp.aref=ones(1,nref); % Constant amplitude pulse
pp.tfp=((techo-ref_len*pi)/2)*pp.T_90/(pi/2); % Free precession time

results=cell(numiter,1);
parfor count = 1:numiter
    disp(['Starting run ' num2str(count)]); 
    pp_curr=pp;
    pp_curr.pref=2*pi*rand(1,nref);
    
    % Run optimization
    [out]=opt_ref_pulse_matched(sp,pp_curr);
    
    % Create results structure
    results_curr=cell(1,7);
    results_curr{1}=[pp_curr.tfp out.tref pp_curr.tfp]; % Store refocusing cycle
    results_curr{2}=[0 out.pref 0];
    results_curr{3}=[0 out.aref 0];
    results_curr{4}=out.axis_rms;
    results_curr{5}=out.sp;
    results_curr{6}=out.pp;
    
    results{count}=results_curr;
    
    disp(out.axis_rms)
end

% Save optimization results
save(file,'results');