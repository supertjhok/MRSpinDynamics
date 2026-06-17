% Repeated optimization runs, store current maximum
% lp -> linearly polarized transmitter

function opt_ref_pulse_circsim_lp_1_repeat(ref_len,techo,file)

[sp,pp]=set_params_lp_tapped;
sp.cycle=2*pi/sp.w0; % Length of one RF cycle
techo = round(techo*(pp.T_90/(pi/2))/sp.cycle)*sp.cycle*(pi/2/pp.T_90);

% Parameters of the refocusing pulse
% ref_len=24/16;
lref=sp.cycle*(pi/2)/pp.T_90; % Each segment is 1 RF cycle long
nref=round(ref_len*pi/lref); % Pulse is ref_len*T_180 long 
params.tref=lref*ones(1,nref);
params.tfp=(techo-pi)/2; % Free precession time
tacq=4*pi; % Acquisition time
params.tacq=round(tacq*(pp.T_90/(pi/2))/sp.cycle)*sp.cycle*(pi/2/pp.T_90);

params.Rs=[sp.Ron 1e6 sp.Roff]; % Source resistances (Q switch on, Q switch off, Tx on)
params.tqs = 1*sp.cycle*(pi/2)/pp.T_90; % Delay between pulse end & Q-switch turn on = 1 RF cycle

count=1;
results={};
for count = 1:20
    params.pref=2*pi*rand(1,nref);
    
    % Run optimization
    [out]=opt_ref_pulse_circsim_lp_1(params,sp,pp);
    
    results{count,1}=[params.tfp out.tref params.tfp]; % Store refocusing cycle
    results{count,2}=[0 out.pref 0];
    results{count,3}=[0 out.aref 0];
    results{count,4}=out.axis_rms;
    results{count,5}=out.params;
    results{count,6}=out.sp;
    results{count,7}=out.pp;
%     save(['results_ref_pulse_lowPower_Sep17_' num2str(nref) '.mat'], 'results');
    save(file,'results');    
    disp(out.axis_rms)
    disp(count)
    
    % Get new initial conditions and repeat optimization
    count=count+1;
end