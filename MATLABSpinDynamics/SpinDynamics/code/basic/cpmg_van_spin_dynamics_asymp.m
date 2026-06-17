% Asymptotic CPMG echo shape using effective rotation axis, excitation and
% refocusing pulses can have different power levels
% ------------------------------------------
% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% NE = number of echoes
% T_FP = free precession time

function [echo_pk,echo_rms]=cpmg_van_spin_dynamics_asymp(texc,pexc,tref,pref,T_90,T_180,T_FP,delt)

delt=delt/1e6;
nref=length(tref);
nexc=length(texc);
tf=zeros(1,nexc);

if nref>1
    neff=calc_rot_axis(T_180/2,tref,pref,[T_FP/2,zeros(1,nref-1),T_FP/2]);
else
    neff=calc_rot_axis(T_180/2,tref,pref,[T_FP/2,T_FP/2]);
end
[echo,tvect]=sim_spin_dynamics_asymp(T_90,T_180,texc,pexc,tf,neff);

ind=find(abs(tvect)<delt/2);

echo_pk=max(abs(echo(ind)));
echo_rms=trapz(tvect(ind),abs(echo(ind)).^2)/length(ind);