function [masy,del_w,tvect,echo]=asy_new_circ(texc,pexc,aexc,tref,pref,aref,T_E,t_acq,del_wmax)

T_FP=T_E-sum(tref);

% Calculate effective rotation axis of refocusing cycle
[neff,del_w]=calc_rot_axis_arba_circ([T_FP/2 tref T_FP/2],[0 pref 0],[0 aref 0],del_wmax);
numpts=length(del_w);

% ----------------------------------------------------------------------
% Calculate asymptotic magnetization and echo waveform
% Use 2-step phase cycling to cancel z-magnetization
% Instead of phase cycling, one can also modify the
% sim_spin_dynamics function so that contributions from z-magnetization are
% ignored, i.e., instead of dot(neff,m)*neff, use ny*my*ny to calculate
% the output spectrum (trans)
% ----------------------------------------------------------------------
if length(texc)==1  % Rectangular, apply timing correction
    [tmp1,tmp3,~]=sim_spin_dynamics_asymp_arba([texc -1/aexc(1)],...
        [pexc 0],[aexc 0],neff,del_w,t_acq);
    [tmp2,tmp4,tvect]=sim_spin_dynamics_asymp_arba([texc -1/aexc(1)],...
        [pexc 0]+pi,[aexc 0],neff,del_w,t_acq);
    my=(tmp1-tmp2)/2;
    echo=(tmp3-tmp4)/2;
else
    [tmp1,tmp3,~]=sim_spin_dynamics_asymp_arba(texc,pexc,aexc,neff,del_w,t_acq);
    if length(texc)==1e2  % Van's excitation pulse, use phase inversion instead of cycling
        [tmp2,tmp4,tvect]=sim_spin_dynamics_asymp_arba(texc,-pexc,aexc,neff,del_w,t_acq);
    else
        [tmp2,tmp4,tvect]=sim_spin_dynamics_asymp_arba(texc,pexc+pi,aexc,neff,del_w,t_acq);
    end
    my=(tmp1-tmp2)/2;
    echo=(tmp3-tmp4)/2;
end

% window function for acquisition only between the 180 pulses
window = sinc(del_w*t_acq/(2*pi));
fy = conv(my,window);
if numpts==2*round(numpts/2); % numpts is even
    masy = fy((numpts/2+1:3*numpts/2))./sum(window);
else
    masy = fy(((numpts+1)/2:3*(numpts-1)/2+1))./sum(window);
end

figure(1);
plot(tvect,abs(echo),'b-'); hold on;
xlabel('Time, t');
ylabel('|Asymptotic echo|');

figure(2);
plot(del_w,abs(my),'b-'); hold on;
plot(del_w,abs(masy),'r-');
xlabel('\omega_{0} / \omega_{1}');
ylabel('|M_{asy}|');