% /dsk/670/hurliman/pulsecraft/COMPOSITES/comp_axis_amex.m
%
% This subroutine calculates the effect of a series of pulses.
% The pulses can be amplitude and /or phase modulated.
% W0 and W1 are the net carrier frequency and the maximum rf amplitude
%
% It needs the following inputs:
%    W0 :        offset of Larmor frequency relative to rf frequency
%    W1 :        effective rf amplitude
%         
%    pulsedur : Vector of pulse duration, in units of nominal 180 pulse.
%    pulseamp  : Vector of pulse amplitude, in units of max. rf amplitude.
%    pulsephase: Vector of phases of pulse, in degrees. 
%       
%    
% Output:
%        nx, ny, nz: components of the net axis of the composite
%                    pulse
%
%  M. Hurlimann, SDR October 1998.
%

%  This sets the free precession time. Echo spacing is given by 
%   tE = 2tcp + pulseduration; tcp in units of t_180/pi.
%tcp = pi*8.90;
%tcp = pi*3;
tcp = pi.*(7 - sum(pulsedur))./2;
%tcp = 9;

ctcp =  ones(size(W0)).*cos(W0.*tcp./2);
stcp = ones(size(W0)).*sin(W0.*tcp./2);

cj  = ctcp;
sjx = zeros(size(W0));
sjy = zeros(size(W0));
sjz = stcp;


for ipulse = 1:length(pulsedur);

    tpulse = pulsedur(ipulse).*pi;
    Omega_v = sqrt( W0.^2 + (pulseamp(ipulse).*W1).^2 );
    cp  = cos(Omega_v.*tpulse./2);
    sp  = sin(Omega_v.*tpulse./2);
    spx = sp.*W1.*pulseamp(ipulse)./Omega_v .* cos(pulsephase(ipulse)*pi/180.);
    spy = sp.*W1.*pulseamp(ipulse)./Omega_v .* sin(pulsephase(ipulse)*pi/180.);
    spz = sp.*W0./Omega_v;
    
    cn  = cj.*cp - spx.*sjx - spy.*sjy - spz.*sjz;
    snx = cp.*sjx + cj.*spx - (sjy.*spz - sjz.*spy);
    sny = cp.*sjy + cj.*spy + (sjx.*spz - sjz.*spx);
    snz = cp.*sjz + cj.*spz - (sjx.*spy - sjy.*spx);
    
    cj  = cn;
    sjx = snx;
    sjy = sny;
    sjz = snz;
    
end;

cn  = cj.*ctcp - sjz.*stcp;
snx = ctcp.*sjx  - (sjy.*stcp );
sny = ctcp.*sjy  + (sjx.*stcp );
snz = ctcp.*sjz + cj.*stcp ;

magsn = max(sqrt(snx.^2 + sny.^2 + snz.^2),1.e-18);
%signy = sign(sny);
rotx = snx ./ magsn ;
roty = sny ./ magsn ;
rotz = snz ./ magsn ;  




    