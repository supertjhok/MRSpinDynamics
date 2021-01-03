% Re-optimize OCT excitation pulse for arbitrary refocusing cycle

function [out]=oct_optimize_pref(pulse_num)

tref=pi*[3 1/7 5/7 1/7 3]; pref=pi*[0 1 0 1 0]; aref=[0 1 1 1 0];
%tref=pi*[3 0.14 0.72 0.14 3]; pref=pi*[0 1 0 1 0]; aref=[0 1 1 1 0];
len_acq=3*pi;

T_90=pi/2; % normalized
delt=0.01*T_90;
tvect=-len_acq/2:delt:len_acq/2;
echo=zeros(1,length(tvect));

tmp=load('dat_files\results_mag_all.mat');
results_sort=tmp.results_sort;
texc=results_sort{pulse_num,1}; pexc=results_sort{pulse_num,2};

[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Recalculate refocusing axis

start=pexc;
nexc=length(texc);
aexc=ones(1,nexc);
lb=zeros(1,nexc); % Lower bound
ub=2*pi*ones(1,nexc); % Upper bound

options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',10000);
pexc=fmincon(@(params)fit_function(params,texc,aexc,neff,del_w,len_acq),start,[],[],[],[],lb,ub,[],options);
[masy]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);

for i=1:length(tvect)
    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
end

out.texc=texc;
out.pexc=pexc;
out.tref=tref;
out.pref=pref;
out.aref=aref;
out.echo_pk=max(abs(echo));
out.echo_rms=sqrt(trapz(tvect,abs(echo).^2));