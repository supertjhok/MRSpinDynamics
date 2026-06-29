function sim_exc_pulse(alpha)

delf=0.1;
delt=alpha/delf;

nvect=13;

i=sqrt(-1);

tvect=linspace(0,2*nvect/delf,20*nvect/delf);
sig=zeros(1,length(tvect));

for j=1:nvect
    ind=find(tvect>delt*(j-1));
    sig(ind)=sig(ind)+exp(i*2*pi*(j-1)*delf*(tvect(ind)-min(tvect(ind))));
end

close all;
figure(1);
subplot(2,1,1);
plot(tvect,abs(sig));
subplot(2,1,2);
plot(tvect,180*phase(sig)/pi);