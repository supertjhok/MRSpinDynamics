 fs = 1e3;                   % sample freq 1kHz
 D = 0 : 1/fs : 2;           % pulse delay times
 t = 0 : 1/fs : 20;          % signal evaluation time
 f1=1;
 f2=10;
 a=(f2-f1)/(t(end)-t(1));
 f=f1+a*t;
 x=square(2*pi*f.*t);
 figure; plot(t,x);
 axis([0 20 -2 2]);
 title('Train chirp pulse');