%  /dsk/670/hurliman/pulsecraft/DIFFUSION/PATHWAYS/asy.m

% Calculate the spectrum of the asymptotic CPMG echo, convoluted with the
% appropriate window function for detection between the 180 pulses.

    w_0    = -50:0.01:50;
    w_1    = 1;
    Omega = sqrt(w_1.*w_1 + w_0.*w_0);
    
    tE    = 15.*pi;
%    tE    =  7.*pi;
    
    b1    = w_0.*tE./2;
    b2    = Omega.*pi./2;
 
    ny = w_1./Omega.*(sin(b2))./sqrt( (w_1./Omega.*sin(b2)).^2 + ...
                 (sin(b1).*cos(b2) + w_0./Omega.*cos(b1).*sin(b2)).^2 );
                 
    ux = -w_0.*w_1./Omega.^2 .*(1-cos(Omega.*pi./2));                
    uy =      w_1./Omega.*sin(Omega.*pi./2);
 
 
  % use proper timing between 90 and 180 pulse
    vy = uy.*cos(w_0) -sin(w_0).*ux;
 
  % asymptotic magnetization
    my = vy.*ny.*ny;
 
  % window function for acquisition only between the 180 pulses
    window = sinc(w_0./pi.*(tE-pi)./2);
     
    fy = conv(my,window);
   
    masyfull = fy(((length(w_0)+1)/2: 3*(length(w_0)-1)/2+1))./sum(window);
   
  % parse it down to a more limited and sparser grid of w0 = [-10:0.02:10] 
  
    omega0 = -10:0.02:10;
    masy = masyfull(4001:2:6001);