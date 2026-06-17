function display_pulse_phase(phi)

nseg=length(phi);

for i=1:nseg
    line([i-1 i],phi(i)*[1 1]); hold on;
    if i<nseg
        if phi(i)~=phi(i+1)
            line([i i],[phi(i) phi(i+1)]);
        end
    end
end

xlim([0 nseg]);
ylim([min(phi)-0.1 max(phi)+0.1]);