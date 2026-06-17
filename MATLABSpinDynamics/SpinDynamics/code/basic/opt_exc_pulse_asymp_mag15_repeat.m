% Repeated optimization runs, store current maximum
% Use longer refocusing pulses
% Use Colm's optimization code for greater speed

function opt_exc_pulse_asymp_mag15_repeat(refpulse_num,tacq)

% Load the refocusing pulse
filname=['LongerRefocusingPulses\Matlab_codes\Pulses\Axis_plus_antisymmetry\PASym_'...
    num2str(refpulse_num) '_1.mat'];
tmp=load(filname); APulse=tmp.APulse; pulse=APulse.pulse;
tmp=size(pulse); nref=tmp(1);
tref=0.1*pi*ones(1,nref); % Segment length = 0.1 x T_180
pref=atan2(pulse(:,3),pulse(:,2))';
aref=sqrt(pulse(:,2).^2+pulse(:,3).^2)'./(2*pi*5e3); % Normalize to w1=1
tref=[3*pi tref 3*pi]; pref=[0 pref 0]; aref=[0 aref 0]; % Add free precession intervals

% Uniform distribution of resonant offsets del_w0 (constant gradient)
maxoffs=20; numpts=2001;
del_w=linspace(-maxoffs,maxoffs,numpts);

% Calculate refocusing axis
[neff]=calc_rot_axis_arba2(tref,pref,aref,del_w,0);
params.neff=neff;
params.del_w=del_w;
params.tacq=tacq;

% Parameters of the excitation pulse
nexc=180; lexc=0.08*pi;
texc=lexc*ones(1,nexc); aexc=ones(1,nexc);

count=1;
results={};
while(1)
    % Try to find good initial conditions
%     for j=1:5e2
%         pexc=2*pi*rand(1,nexc);
%         [masy]=cpmg_van_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,del_w,tacq);
%         echo_ms=trapz(del_w,abs(masy).^2);
%         if j==1
%             echo_ms_max=echo_ms;
%             pexc_max=pexc;
%         else
%             if echo_ms>echo_ms_max
%                 disp(j)
%                 echo_ms_max=echo_ms
%                 pexc_max=pexc;
%             end
%         end
%     end
%     
%     params.texc=texc;
%     params.pexc=pexc_max;
    
    params.texc=texc;
    if count==1
        params.pexc=2*pi*rand(1,nexc);
    else
        params.pexc=0.9*pexc0+0.1*2*pi*rand(1,nexc);
    end
    
    % Run optimization
    [out]=opt_exc_pulse_asymp_mag15(params);
    
    results{count,1}=out.texc;
    results{count,2}=out.pexc;
    results{count,3}=tref;
    results{count,4}=pref;
    results{count,5}=out.echo_pk;
    results{count,6}=out.echo_rms;
    save(['results_mag15_' num2str(refpulse_num) '.mat'], 'results');
    
    if (count==1) || (out.echo_rms>echo_rms_max)
        echo_rms_max=out.echo_rms;
        pexc0=out.pexc;
    end
        
    disp(out.echo_pk)
    disp(out.echo_rms)
    disp(count)
    
    % Get new initial conditions and repeat optimization
    count=count+1;
end