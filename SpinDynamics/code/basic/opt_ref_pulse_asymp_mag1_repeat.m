% Repeated optimization runs, store current maximum

function opt_ref_pulse_asymp_mag1_repeat(tacq)

% Uniform distribution of resonant offsets del_w0 (constant gradient)
maxoffs=20; numpts=2001;
params.del_w=linspace(-maxoffs,maxoffs,numpts);
params.opt_window=hamming(numpts)';
params.tacq=tacq;

% Parameters of the refocusing pulse, assuming internal anti-symmetry
nref=13; lref=0.08*pi;
tref=lref*ones(1,nref);
params.tfp=3*pi; % Free precession time

count=1;
results={};
while(1)
    params.tref=tref;
    if count==1
        params.pref=2*pi*rand(1,nref);
    else
       % params.pref=0.9*pref0+0.1*2*pi*rand(1,nref);
        params.pref=2*pi*rand(1,nref);
    end
    
    % Run optimization
    [out]=opt_ref_pulse_asymp_mag1(params);
    
    results{count,1}=[params.tfp out.tref out.tref params.tfp]; % Store refocusing cycle
    results{count,2}=[0 out.pref -fliplr(out.pref) 0];
    results{count,3}=[0 out.aref out.aref 0];
    results{count,4}=out.axis_rms;
    save(['results_ref_mag1_' num2str(nref) '.mat'], 'results');
    
    if (count==1) || (out.axis_rms>axis_rms_max)
        axis_rms_max=out.axis_rms;
        pref0=out.pref;
    end
        
    disp(out.axis_rms)
    disp(count)
    
    % Get new initial conditions and repeat optimization
    count=count+1;
end