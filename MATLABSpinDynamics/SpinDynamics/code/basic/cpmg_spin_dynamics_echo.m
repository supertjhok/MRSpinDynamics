% NE - number of echoes
% TE - echo spacing
% rat - T90/T180 pulse duration ratio
% Delays follow pulses

function [echo,tvect]=cpmg_spin_dynamics_echo(T_90,rat,NE,TE,T1,T2)

T_180=T_90*rat;

tp=T_180*ones(1,NE+1);
tp(1)=T_90;

phi=(pi/2)*ones(1,NE+1);
phi(1)=0;

tf=(TE-T_180)*ones(1,NE+1);
% Martin's correction for gradient fields - gives more peak signal by making SE
% and DE echoes occur closer together in time
tf(1)=0.5*(TE-T_180-T_180/pi);
tf(NE+1)=0.5*(TE-T_180);

% No PAP
[echo,tvect]=sim_spin_dynamics_allpw(T_90,tp,phi,tf,T1,T2);