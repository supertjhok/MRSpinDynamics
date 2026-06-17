% Generate pulse shape file for Kea from variable time
% segment data
% amplitude -> normalized to 1, phase -> in radians
% ntot -> number of segments

function gen_shape_kea(filname,ntot,timing,amplitude,phase)

power=amplitude;

sign=zeros(1,ntot);
sign(abs(phase)>(pi/2))=8192;

% Open file
filname_path=['wave/' filname];
fhandle1=fopen(filname_path,'w');

% Calculate segment lengths, correct for round-off errors
nseg=round(ntot*timing/sum(timing));
tmp=sum(nseg);
nseg(length(power))=nseg(length(power))+(ntot-tmp);

% Write (power,sign) pairs
for i=1:length(power)
    for j=1:nseg(i)
        fprintf(fhandle1,'%s\r\n',[num2str(power(i)) ' ' num2str(sign(i))]);
    end
end

% Close file
fclose(fhandle1);