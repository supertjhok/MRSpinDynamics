% Generate pulse shape file for Kea from constant time segment data
% amplitude -> normalized to 1, phase -> in radians
% nseg -> number of segments

function gen_shape_kea_arbphase(filname,amplitude,phase)

phase=round(16383*phase/(2*pi)); % quantize
nseg=length(amplitude);

% Open file
filname_path=['wave/' filname];
fhandle1=fopen(filname_path,'w');

% Write (amp,phase) pairs
for i=1:nseg
        fprintf(fhandle1,'%s\r\n',[num2str(amplitude(i)) ' ' num2str(phase(i))]);
end

% Close file
fclose(fhandle1);