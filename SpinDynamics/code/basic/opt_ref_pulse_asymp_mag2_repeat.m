% Repeated optimization runs, store current maximum

function opt_ref_pulse_asymp_mag2_repeat(tacq)

% Uniform distribution of resonant offsets del_w0 (constant gradient)
maxoffs=20; numpts=2001;
params.del_w=linspace(-maxoffs,maxoffs,numpts);
params.opt_window=hamming(numpts)';
params.tacq=tacq;

% Parameters of the refocusing pulse, assuming internal anti-symmetry
nref=36; lref=0.08*pi;
tref=lref*ones(1,nref);
params.tfp=6*pi; % Free precession time

% Try to find good initial conditions by using results from a previous
% optimization
%tmp=load('results_ref_mag2\tseg_0.1xt_180\results_ref_mag2_20_run1'); old=tmp.results;
%nold=20; sizold=size(old); old_num=5;

count=1;
results={};
while(1)
    params.tref=tref;
    if count==1
        params.pref=2*pi*rand(1,nref);
        
        %old_num=ceil(rand(1)*sizold(1))
        %pold=old{old_num,2};
        %params.pref=[2*pi*rand(1,nref-nold) pold(2:nold+1)];
    else
        %params.pref=0.9*pref0+0.1*2*pi*rand(1,nref);
        params.pref=2*pi*rand(1,nref);
        
        %old_num=ceil(rand(1)*sizold(1))
        %pold=old{old_num,2};
        %params.pref=[2*pi*rand(1,nref-nold) pold(2:nold+1)];
    end
    
    % Run optimization
    [out]=opt_ref_pulse_asymp_mag2(params);
    tref=out.tref; pref=out.pref; aref=out.aref;
    
    results{count,1}=[params.tfp/2 tref tref(1:nref-1) params.tfp/2]; % Store refocusing cycle
    results{count,2}=[0 pref -fliplr(pref(1:nref-1)) 0];
    results{count,3}=[0 aref aref(1:nref-1) 0];
    results{count,4}=out.axis_rms;
    save(['results_ref_mag2_' num2str(nref) '.mat'], 'results');
    
    if (count==1) || (out.axis_rms>axis_rms_max)
        axis_rms_max=out.axis_rms;
        pref0=out.pref;
    end
        
    disp(out.axis_rms)
    disp(count)
    
    % Get new initial conditions and repeat optimization
    count=count+1;
end