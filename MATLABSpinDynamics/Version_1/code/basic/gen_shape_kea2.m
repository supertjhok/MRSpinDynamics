% Generate pulse shape file for Kea2 from variable time
% segment data
% amplitude -> normalized to 1, phase -> in degrees
% ntot -> number of segments

function gen_shape_kea2(filname,ntot,timing,amplitude,phase)

phase=mod(phase,360); % Phase modulo 2*pi (360 degrees)

% Open file
%filname_path=['wave/' filname]; % Unix
filname_path=['wave\' filname]; % Windows
fhandle1=fopen(filname_path,'w');

% Calculate segment lengths, correct for round-off errors
nseg=round(ntot*timing/sum(timing));
tmp=sum(nseg);
nseg(length(amplitude))=nseg(length(amplitude))+(ntot-tmp);

% Write (amplitude,phase) pairs
for i=1:length(amplitude)
    for j=1:nseg(i)
        fprintf(fhandle1,'%s\r\n',[num2str(amplitude(i)) ' ' num2str(phase(i))]);
    end
end

% Close file
fclose(fhandle1);