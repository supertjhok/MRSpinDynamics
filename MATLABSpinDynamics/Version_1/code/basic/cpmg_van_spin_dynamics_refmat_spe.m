% texc,pexc = excitation pulse times, phases
% TE = echo spacing
% T_ER = free precession time between excitation and first refocusing pulse
% tpar = integration time parameters
% tpar(1), tpar(2) % SPE minimum and maximum times (SPE begins at tmin)
% tpar(3), tpar(4) % Echo minimum and maximum times (Echo centered at
% zero)
% Delays follow pulses

function [spe_pk,spe_rms,echo_pk,echo_rms]=cpmg_van_spin_dynamics_refmat_spe(texc,pexc,T_90,T_ER,refmat,tpar)

tpar=tpar/1e6;

nexc=length(texc);

tf=zeros(1,nexc);
tf(nexc)=T_ER;

[spe,tvect_spe,echo,tvect]=sim_spin_dynamics_refmat_spe(T_90,texc,pexc,tf,refmat);

ind=find(tvect_spe>tpar(1) & tvect_spe<tpar(2));
spe_pk=max(abs(spe(ind)));
spe_rms=trapz(tvect_spe(ind),abs(spe(ind)).^2)/(tpar(2)-tpar(1));

ind=find(tvect>tpar(3) & tvect<tpar(4));
echo_pk=max(abs(echo(ind)));
echo_rms=trapz(tvect(ind),abs(echo(ind)).^2)/(tpar(4)-tpar(3));