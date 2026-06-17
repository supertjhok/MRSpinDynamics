
function [phiq]=quantize_phase(phi,sp,pp)

numpts=length(phi);
Np=pp.NumPhases; 
%w=pp.w;
%tau=sp.L/(sp.R+pp.Rsref(1)); % Assume Rsref is constant
%theta=-atan2(w*tau,1);

% Quantize phase to one of NumPhases evenly-spaced values 
pq=(2*pi/Np)*linspace(0,Np-1,Np);

phiq=zeros(1,numpts);
for i=1:numpts
    [~,ind]=min(abs(pq-phi(i)));
    phiq(i)=pq(ind);
end
% phiq=phiq-phiq(1); % Phase of first segment must be zero (or pi)

