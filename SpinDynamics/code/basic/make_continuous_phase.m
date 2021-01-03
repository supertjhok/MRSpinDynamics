% Unwrap phase of multi-segment RF pulses
% tseg_vect = segment lengths, ph_vect = phases (radians)

function [ph_vect_cont,del_w]=make_continuous_phase(tseg_vect,ph_vect)

numph=length(ph_vect);
ph_vect_cont=ph_vect;
seg_vect=linspace(1,numph,numph);

% Calculate unwrapped phase
for i=1:numph-1
    dff=abs(ph_vect_cont(i)-ph_vect_cont(i+1));
    jval=0;
    for j=-2:2
        tmp=abs(ph_vect_cont(i)-ph_vect_cont(i+1)-2*pi*j);
        if tmp<dff
            jval=j;
            dff=tmp;
        end
    end
    if jval
        ph_vect_cont(i+1)=ph_vect_cont(i+1)+2*pi*jval;
    end
end

% Calculate offset frequency
seg_vectc=(seg_vect(1:numph-1)+seg_vect(2:numph))/2;
tseg_vectc=(tseg_vect(1:numph-1)+tseg_vect(2:numph))/2;
del_w=diff(ph_vect_cont)./tseg_vectc;

figure(1);
subplot(2,1,1);
plot(seg_vect,ph_vect/(2*pi),'ko-'); hold on;
plot(seg_vect,ph_vect_cont/(2*pi),'rx-');
ylabel('Phase (cycles)');

subplot(2,1,2);
plot(seg_vectc,del_w,'ro-'); hold on;
xlabel('Segment number');
ylabel('\Delta\omega/\omega_{1}'); % Assumes omega_1 = 1



