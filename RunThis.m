%% Example script to run spectral power dual regression
%
%
% Olivia Viessmann <oviessmann@mgh.harvard.edu>
%                  12/09/2017
%
%


%% *********** Step 0 : Set up necessary parameters for the run **********

%Cut-off frequency of GLM 
fminFreqGLM = 0.2;
%TR in seconds
TR = 0.328;
%Sampling rate of physiological recording in Hz
Samplefreq      = 1000;
TRinSamplefreq  = TR*Samplefreq;  %We will need TR in units of physiological sampling 

%P-Value for GLM
pVal =0.05;

%Pve threshold for masking
PveThres = 0.7;

%Normalisation of spectra - yes(1) or no(0)
NormFlag = 1;

%Convergence level for dual GLM
ConvLevel = 0.1;


%% ****************** STEP 1 Load physiological recording ****************
PhysioDir       = 'physio.txt';
PhysioFileID    = fopen(char(PhysioDir));
A               = fscanf(PhysioFileID,'%f %f %f',[5 Inf]);
%Columns:
% 1  - time, 2 - respiration, 3 - cardiac (pulse ox), 4 - oxygen saturation
% 5  - RF trigger
% Find start and end of scan by detecting trigger events
Triggers    = find(A(5,:)>3);
Scanstart   = min(Triggers); 
Scanend     = max(Triggers)+TRinSamplefreq-10; %trigger itself is 10 bins on recording (10 ms trigger duration)

%The first 30 volumes were thrown away (~10 secs)
PhysioTrace = A(:,Scanstart+30*TRinSamplefreq:Scanend);

%Subsample to TR time
PhysioTraceDownSampled = PhysioTrace(:,1:TRinSamplefreq:end);
    
%Calculate the spectral resolution (for plots etc.)
SamplefreqTR = 1/TR;
PrecisionTR = SamplefreqTR/(size(PhysioTraceDownSampled,2)+1);
fTR = 0:PrecisionTR:SamplefreqTR/2-PrecisionTR;

%Create spectra via Fourier transform, throw away the DC component and
%normalise the spectra (easier for plotting)
SpecCardTR = abs(fft(PhysioTraceDownSampled(3,:))); SpecCardTR = SpecCardTR(2:size(fTR,2));
SpecCardTR = SpecCardTR./sum(SpecCardTR); 
SpecRespTR = abs(fft(PhysioTraceDownSampled(2,:))); SpecRespTR = SpecRespTR(2:size(fTR,2));
SpecRespTR = SpecRespTR./sum(SpecRespTR);
    
%Plot spectra to get a first impression
figure
plot(fTR(2:end), SpecCardTR, fTR(2:end), SpecRespTR); 
legend('Cardiac spectrum', 'Respiratory spectrum');title('External physiological recording');
xlabel('Frequency [Hz]'); ylabel('Normalised power');

%% ******* Step 2: Get the tissue masks and functional data **************
% Get tissue pve maps and epi time series data. 
load('TissuePve.mat');
% Load pve maps and threshold by minimum pve value. These maps are the PVE
% maps from the T1 image, transformed already to the EPI
GreyMask    = TissuePve.Grey;
WhiteMask   = TissuePve.White;
GreyMask(GreyMask<PveThres)   = 0;
WhiteMask(WhiteMask<PveThres) = 0;
% Load the time series data. These are already motion corrected and detrendended.
% Data set needs to be loaded in segments due to github upload restrictionsin on fie sizes.
% Skip the first 30 TRs
tmp = dir('Rsfmri_residuals*');
names = {tmp.name};

for i = 1:length({tmp.name})
    load(char(names(i)));
end
Residuals = cat(4,Rsfmri_residuals1to100, Rsfmri_residuals101to200, ...
                 Rsfmri_residuals201to300, Rsfmri_residuals301to400, ...
                 Rsfmri_residuals401to500, Rsfmri_residuals501to600, ...
                 Rsfmri_residuals601to700, Rsfmri_residuals701to800, ...
                 Rsfmri_residuals801to900, Rsfmri_residuals901to1000, ...
                 Rsfmri_residuals1001to1100, Rsfmri_residuals1101to1200, ...
                 Rsfmri_residuals1201to1300, Rsfmri_residuals1301to1400, ...
                 Rsfmri_residuals1401to1500, Rsfmri_residuals1501to1600, ...
                 Rsfmri_residuals1601to1700, Rsfmri_residuals1701to1800, ...
                 Rsfmri_residuals1801to1900, Rsfmri_residuals1901to2000, ...
                 Rsfmri_residuals2001to2100, Rsfmri_residuals2101to2200);
%Remove the first 30 TRs
Residuals = Residuals(:,:,:,30:end);

%% ******** Step 3: Let's start the dual regressions *********************
 MaskList = {'GreyMask' 'WhiteMask'}
 tic
for c = 1:numel(MaskList)
        disp(['Running dual regression in mask: ' MaskList{c}]);
        Mask = eval(MaskList{c});
        
        %Only run in mask of intrest
        BinMask = Mask; BinMask(BinMask>0) = 1; BinMask(BinMask<1) = 0;
        ResidsInMask = bsxfun(@times, Residuals, BinMask);
        %Run with external as initial guess and a free version with
        %data-driven exploration of the spectra
        %a) run the informed variant first
        [AVoxelSpectraInfo, SBetasFirst,BetasRefinedInfo, RefinedSpecInfo] = DualRegressionLoop(ResidsInMask,fTR,fminFreqGLM,SpecCardTR,SpecRespTR,pVal,NormFlag,ConvLevel);       
        %b) now run the data-driven one
        [AVoxelSpectraFree, SBetasFirst,BetasRefinedFree, RefinedSpecFree] = DualRegressionLoop(ResidsInMask,fTR,fminFreqGLM,[],[],pVal,NormFlag,ConvLevel);
        
        %Store
        CardiacPEInfo(:,:,:,c) = squeeze(BetasRefinedInfo(:,:,:,2));
        CardiacPEFree(:,:,:,c) = squeeze(BetasRefinedFree(:,:,:,2));
        
        close all
end
    toc

figure
imagesc(squeeze(CardiacPEInfo(:,:,16,2)))
title('cardiac PE map for informed dual regression, WM')
figure 
imagesc(squeeze(CardiacPEFree(:,:,16,2)))
title('cardiac PE map for free dual regression, WM')