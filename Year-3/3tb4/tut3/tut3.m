% Read the .wav file (replace file_name by your group’s file
% Variable x stores the wave and fs stores the sampling rate
[x, fs]=audioread('C:\Shared\alex-bartella-schoolwork\3tb4\tut3\L02_Group11.wav');
% Perform FFT on the original signal to determine the frequency of the \noise"
L=length(x);
NFFT=2^nextpow2(L);
X=fft(x,NFFT)/fs;

% Show the sampling rate
fs
% We know the sampling rate is 8000
% We need now to plot our FFT to find the source of the noise.
% Plot single-sided amplitude spectrum
f=fs/2*linspace(0,1,NFFT/2+1);
plot(f,2*abs(X(1:NFFT/2+1)));

% Reading the FFT we realize that the frequency we want to remove is 2000
% (Hint: our noise is a pure sine wave)
% Now specify the frequency you want to eliminate by setting:
fkill=1/2 ;
% Hint: fkill is always in the range of 0 { 1, and is normalized to frequency of fs/2.
% Determine the coefficients of the FIR filter that will remove that frequency.
% Start off the following blank with the value 4, to numbers larger than 160.
% Note: the following filter only works with EVEN numbers.
%min=inf;
%for j=4:2:160
    coeff=firgr(60,[0,fkill-0.1, fkill, fkill+0.1, 1], [1,1,0,1,1],{'n','n','s','n','n'});

%end
% Plot the filter
% Plot the frequency response of the designed filter to verify that it satisfies the
% Requirements
freqz(coeff,1);
% You should try different filter lengths in the firgr command and find out which one is
% the best. Filter length of 4 is terrible. Ideally, your filter should only filter out the
% noise while passing all other signals. Try increasing your filter length until you can
% achieve an adequate result.
% Be sure to plot (with freqz()) each time you create a new filter. If you pick a filter
% length too large, the filter will \blow up". If you are unsure whether your filter has
% blown up or not, seek help from a TA.
coeff*32768
% Save these coefficients in a text file, You will need them when coding the FIR filter.
fid=fopen('C:\Shared\alex-bartella-schoolwork\3tb4\tut3\audiofile.txt','w');
% If you make a typing error with the following for-end block, you need to start from the
%\for" line again.

for i=1:length(coeff)
    fprintf(fid,'coeff[%3.0f]=%10.0f;\n',i-1,32768*coeff(i));
end
fclose(fid);
% Filter the input signal x(t) using the designed FIR filter to get y(t).
y=filtfilt(coeff,1,x);
% Perform FFT on the filtered signal to observe the absence of frequency of the \noise".
Y=fft(y,NFFT)/L;
% Play the unfiltered sound (your system must have a
% working speaker or headphone)
% Multiply by 3 to make the volume 3 times louder.
%sound(3*x,fs);
% Pause 5 seconds. (this is only necessary if you run these commands as a script)
pause(5);
% Play the filtered sound
%sound(3*y,fs);
% The secret code for your group is _____________________
% Create two plots to compare
subplot(2,1,1);
% The first plot shows the FFT of the original signal.
plot(f,2*abs(X(1:NFFT/2+1)));
xlabel('frequency (Hz)');
ylabel('|X(f)|');
title("This one")
% The second plot shows the FFT of the filtered signal.
subplot(2,1,2);
plot(f, 2*abs(Y(1:NFFT/2+1)));
xlabel('frequency(Hz)');
% Write the filtered audio file to disk.

audiowrite('C:\Shared\alex-bartella-schoolwork\3tb4\tut3\filtered.wav',y,fs)