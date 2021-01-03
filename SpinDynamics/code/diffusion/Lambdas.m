%  /dsk/670/hurliman/pulsecraft/DIFFUSION/PATHWAYS/Lambdas.m

% calculate the matrix elements for the 180_y rf pulses


  omega0 = -10:0.02:10;
  omega1 = 1;
  
  Lambda = zeros(9,length(omega0));
  Linit  = zeros(2,length(omega0));
 

  Omega  = sqrt(omega1.^2 + omega0.^2);
  w0 = omega0./Omega;
  w1 = omega1./Omega;
    

        
  tp = pi;
  cp = cos(Omega.*tp);
  sp = sin(Omega.*tp);
   
   
  pulse = 1i;  % 1 for x - pulse;  i for y -pulse

  Lambda(1,:) = ( w1.^2 + (1+(w0.^2)).*cp ) ./2 - 1i * w0.*sp;   % -1, -1
   
  Lambda(2,:) =   w1./2.*( (w0.*(1-cp)) + 1i* sp ).*pulse ;      % -1,  0
   
  Lambda(3,:) =   w1.^2./2.*(1-cp).*(pulse).^2;                 % -1, +1
   
  Lambda(4,:) =   w1   .*( (w0.*(1-cp)) + 1i* sp ).*pulse' ;     %  0, -1
   
  Lambda(5,:) =   w0.^2 + w1.^2  .*cp;                          %  0,  0
   
  Lambda(6,:) =   w1   .*( (w0.*(1-cp)) - 1i* sp ).*pulse;       %  0, +1
   
  Lambda(7,:) =   w1.^2./2.*(1-cp).*(pulse').^2;                % +1, -1
   
  Lambda(8,:) =   w1./2.*( (w0.*(1-cp)) - 1i* sp ).*pulse' ;     % +1, 0
   
  Lambda(9,:) = ( w1.^2 + (1+(w0.^2)).*cp ) ./2 + 1i * w0.*sp;   % +1, +1   


%  calculate the matrix elements for the initial 90_-x rf pulse, 
%  including the proper timing adjustment

  tp = pi/2;
  cp = cos(Omega.*tp);
  sp = sin(Omega.*tp);
   
  pulse = -1;
   
  Linit(1,:)  = w1   .*( (w0.*(1-cp)) + 1i* sp ).*pulse'.*exp(+1i*omega0) ;  % 0, -1
  Linit(2,:)  = w1   .*( (w0.*(1-cp)) - 1i* sp ).*pulse.*exp(-1i*omega0) ;  % 0, +1
  

  figure(2)
  plot(omega0,real(Linit(1,:)),'b-'); hold on;
  plot(omega0,imag(Linit(1,:)),'r-');