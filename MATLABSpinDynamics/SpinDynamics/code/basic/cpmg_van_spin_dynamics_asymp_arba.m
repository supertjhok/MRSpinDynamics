% Asymptotic CPMG echo shape using effective rotation axis, excitation and
% refocusing pulses can have arbitrary power levels
% ------------------------------------------
% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% NE = number of echoes
% T_FP = free precession time

function [echo_pk,echo_rms,trans,del_w,echo,tvect]=...
    cpmg_van_spin_dynamics_asymp_arba(texc,pexc,aexc,tref,pref,aref,T_FP)

t_acq=T_FP;
delt=t_acq;

% Calculate effective rotation axis of refocusing cycle
[neff,del_w]=calc_rot_axis_arba([T_FP/2 tref T_FP/2],[0 pref 0],[0 aref 0]);

% ----------------------------------------------------------------------
% Calculate asymptotic magnetization and echo waveform
% Use 2-step phase cycling to cancel z-magnetization
% Instead of phase cycling, one can also modify the
% sim_spin_dynamics function so that contributions from z-magnetization are
% ignored, i.e., instead of dot(neff,m)*neff, use ny*my*ny to calculate
% the output spectrum (trans)
% ----------------------------------------------------------------------
if length(texc)==1
    % Apply timing correction
    [tmp1,tmp3,~]=sim_spin_dynamics_asymp_arba([texc -1/aexc(1)],...
        [pexc 0],[aexc 0],neff,del_w,t_acq);
    [tmp2,tmp4,tvect]=sim_spin_dynamics_asymp_arba([texc -1/aexc(1)],...
        [pexc 0]+pi,[aexc 0],neff,del_w,t_acq);
    trans=(tmp1-tmp2)/2;
    echo=(tmp3-tmp4)/2;
else
    [tmp1,tmp3,~]=sim_spin_dynamics_asymp_arba(texc,pexc,aexc,neff,del_w,t_acq);
    [tmp2,tmp4,tvect]=sim_spin_dynamics_asymp_arba(texc,pexc+pi,aexc,neff,del_w,t_acq);
    trans=(tmp1-tmp2)/2;
    echo=(tmp3-tmp4)/2;
end

ind=find(abs(tvect)<delt/2);

echo_pk=max(abs(echo(ind)));
echo_rms=trapz(tvect(ind),abs(echo(ind)).^2)/length(ind);