% Quantize a phase vector to the nearest set of NumPhase values
% Soumyajit Mandal, 01/20/2021

function [phiq]=quantize_phase(phi,sp,pp)

numpts=length(phi);
Np=pp.NumPhases;
%w=pp.w;
%tau=sp.L/(sp.R+pp.Rsref(2)); % Assume Rsref is constant
%theta=-atan2(w*tau,1);

% Quantize phase to one of NumPhases evenly-spaced values 
pq=(2*pi/Np)*linspace(0,Np-1,Np);

phiq=zeros(1,numpts);
for i=1:numpts
    [~,ind]=min(abs(pq-phi(i)));
    phiq(i)=pq(ind);
end
% phiq=phiq-phiq(1); % Phase of first segment must be zero (or pi)

