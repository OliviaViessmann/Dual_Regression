function [ AllVoxelSpectra, SBetasFirstRun, SBetasRefined,RefinedSpec] = DualRegressionLoop( Resids,f,fmin,SpecCard,SpecResp,pVal, NormFlag,ConvLevel)
% This function runs a dual regression of power spectra.
%
%***** INPUT ******
%
% Residuals         - 4D time series data
% fTR               - frequency vector, subsampled to TR
% fmin              - lower GLM frequency threshold, in Hz
% SpecCardTR        - externally measured cardiac spectrum subsampled to TR and
%                     zero component removed
% SpecRespTR        - externally measured breathing spectrum subsampled to TR
%                     and zero component removed
% pVal              - significance level for GLM
% NormFlag          - normalise(1) spectra from lower cut off frequency to
%                     Nyquist or from f =  0 to Nyquist (0)
% ConvLevel         - level of convergence to stop (it's the absolute sum of
%                     differences between the iterated spectra) 
%
% *****OUTPUT ******
%
% AllVoxelSpectra         - 4D matrix of voxel spectra
% SBetasFirstRun          - spatial regressor from first GLM (associated
%                           with externally measured cardio-respiratory
%                           spectra), i.e. if no dual regression iteration
%                           is performed. 
% SBetasRefined           - refined spatial parametere estimate maps from
%                           iterative dual regression runs
% RefinedSpec             - refined spectra after dual regression
%
%***************************************************************************
figure
tic
%Check everything is available
if(isempty (Resids))
    error('No time series. ')
end
if(isempty (f))
    error('No frequency spetrum specified. ')
end
if(isempty (fmin))
    error('No lower GLM frequency threshold specified.')
end
if(isempty (SpecCard))
    disp('No cardiac spectrum specified. Will run the data-driven variant. ')
    FitCardBool = 1;
    SpecCard = ones(size(f(2:end)))./numel(f(2:end)); %uni-distributed, like noise baseline
else 
    FitCardBool = 0;
end
if(isempty (SpecResp))
    disp('No respiratory spectrum specified. Will run the data-driven variant. ');
    FitRespBool = 1;
    SpecResp = ones(size(f(2:end)))./numel(f(2:end)); %uni-distributed, like noise baseline
else
    FitRespBool  =0;
end
if(isempty(NormFlag)|| NormFlag >1 || NormFlag<0)
    NormFlag = 1;
    disp('No normalisation flag specified, default is 1.');
end
if(isempty(ConvLevel))
    Converged = 0.1;
    disp('No convergence level specified, default is set to 0.1');
end

[Num_RO,Num_PE,Slices, Vols] = size(Resids);
disp(['The time series data matrix is ' num2str(Num_RO) ' x ' num2str(Num_PE)  ' x ' num2str(Slices) ' and the number of volumes is ' num2str(Vols)]);

%*******************CALCULATE SPECTRUM*******************************
disp('Starting first step.');
%The index of the  minimum frequency
PrecisionTR = f(2)-f(1); %Calculate the precision from arbitrary indixes in the frequency vector
fminGLMindex = round(fmin/PrecisionTR);

disp('Starting calculation of voxel spectra.');
%Calculate normalised spectrum in each voxel
if NormFlag == 1
    %Normalise only between cut off frequency and Nyquist frequency and cut
    %down residual time series
    AllVoxelSpectra = zeros(Num_RO,Num_PE,Slices,size(f,2)-fminGLMindex);
else
    AllVoxelSpectra = zeros(Num_RO,Num_PE,Slices,size(f,2)-1);
end

%size(AllVoxelSpectra)
parpool('local',2)
parfor i = 1:Num_RO
    for j = 1:Num_PE
        for k = 1:Slices
            if any(Resids(i,j,k,:))
                tmp = squeeze(abs(fft(Resids(i,j,k,:))));
                End = floor(size(tmp,1)/2);
                %size(tmp(2:End))
                if NormFlag == 1
                    AllVoxelSpectra(i,j,k,:) = tmp(fminGLMindex:End)./sum(tmp(fminGLMindex:End));
                elseif NormFlag == 0
                   AllVoxelSpectra(i,j,k,:) = tmp(2:End)./sum(tmp(2:End));
                end
            end
        end
    end
end
disp('Finished calculation of voxel spectra.');
%Cut down to minimum neuro free frequency range
if NormFlag == 1
    AllVoxelSpectraGLM = AllVoxelSpectra;
else
    AllVoxelSpectraGLM = AllVoxelSpectra(:,:,:,fminGLMindex:end);
end
%Iteration counter
Iter = 1;
ConvBool = 0;
while ConvBool == 0
    
    if Iter ==1 
        SpecCardGLM     = SpecCard(fminGLMindex:end); 
        SpecCardGLM     = SpecCardGLM./sum(SpecCardGLM);
        SpecRespGLM     = SpecResp(fminGLMindex:end);
        SpecRespGLM     = SpecRespGLM./sum(SpecRespGLM);
    else
        SpecCardGLM        = RefinedCardSpec';
        SpecRespGLM        = RefinedRespSpec';
    end


    %*******************RUN FIRST GLM**************************************
    if FitRespBool == 1 && FitCardBool == 0 && Iter == 1
        XFirst = SpecCardGLM';
        BetasFirst = zeros(Num_RO,Num_PE,Slices,2);
        PValsFirst = ones(Num_RO,Num_PE,Slices,2);
    elseif FitCardBool == 1 && FitRespBool == 0 && Iter == 1
        XFirst = SpecRespGLM';
        BetasFirst = zeros(Num_RO,Num_PE,Slices,2);
        PValsFirst = ones(Num_RO,Num_PE,Slices,2);
    elseif FitCardBool == 1 && FitRespBool == 1 &&  Iter == 1
        XFirst = SpecCardGLM';
        BetasFirst = zeros(Num_RO,Num_PE,Slices,2);
        PValsFirst = ones(Num_RO,Num_PE,Slices,2);
    else        
        XFirst = [SpecCardGLM; SpecRespGLM]';
        BetasFirst = zeros(Num_RO,Num_PE,Slices,3);
        PValsFirst = ones(Num_RO,Num_PE,Slices,3);
    end
    
    parfor i = 1:Num_RO
        for j = 1:Num_PE
            for k = 1:Slices
                if any(Resids(i,j,k,:))
                    YFirst = squeeze(AllVoxelSpectraGLM(i,j,k,:));
                    [Betastmp, devtmp, stats] = glmfit(XFirst,YFirst);
                    BetasFirst(i,j,k,:) = Betastmp;
                    PValsFirst(i,j,k,:) = stats.p;
                end
            end
        end
    end
    disp('First GLM finished.');

    %Assess first GLM results
    %Multiply significant Betas with masks
    Ptmp = PValsFirst;  Ptmp(Ptmp<pVal) = 2;  Ptmp(Ptmp<2) = 0;  Ptmp(Ptmp>0) = 1;
    SBetasFirst = BetasFirst.*Ptmp;
    SBetasFirst(isnan(SBetasFirst)==1)=0;
   

    %Do not allow negative power amplitudes and set all negative amplitudes
    %to almost zero
    SBetasFirst(SBetasFirst<0) =0;
    
    %Store betas from first run (non-dual regressed for comparison 
    if Iter ==1
        SBetasFirstRun = SBetasFirst;
    end

    % ***************FIRST STEP OF DUAL REG********************************    
    % Create vector of all voxel spectra, stripped along one spatial
    % dimension and along the frequency dimension, i.e. Y[ixjxk,f]
    % Explanatory variables are the spatial maps for cardiac and respiration
    % from first GLM, stripped along one spatial dimension
    
    disp('First step of dual regression is starting.');
    if (Iter ==1) && (FitRespBool == 1 || FitCardBool == 1) 
        tmp1 = SBetasFirst(:,:,:,1); tmp2 = SBetasFirst(:,:,:,2); 
        XDual           = [tmp1(:), tmp2(:)];
        BetasDual       = zeros(size(AllVoxelSpectraGLM,4),2);
        PValsDual       = ones(size(AllVoxelSpectraGLM,4),2);  
    else
        tmp1 = SBetasFirst(:,:,:,1); tmp2 = SBetasFirst(:,:,:,2); 
        tmp3 = SBetasFirst(:,:,:,3);
        XDual           = [tmp1(:), tmp2(:), tmp3(:)];
        BetasDual       = zeros(size(AllVoxelSpectraGLM,4),3);
        PValsDual       = ones(size(AllVoxelSpectraGLM,4),3);  
    end

    
    %Run spatial regression
    parfor i = 1:size(AllVoxelSpectraGLM,4)
        tmpSpatial    = AllVoxelSpectraGLM(:,:,:,i);        %only compare above GLM threshold, as we don't know what happens at lower frequencies
        YDual   = tmpSpatial(:); 
        if any(YDual)
           [Betastmp, devtmp, stats] = glmfit(XDual,YDual,'normal','constant','off');
                    BetasDual(i,:) = Betastmp;
                    PValsDual(i,:) = stats.p;
        end
    end
    disp(['Spatial regression GLM finished for ' num2str(Iter)  ' iterations']);

    %Set up refined spectra
    %Multiply significant Betas with masks
    Ptmp = PValsDual; Ptmp(Ptmp<pVal) = 2; Ptmp(Ptmp<2) = 0; Ptmp(Ptmp>0) = 1;
    SignificantBetasDual = BetasDual.*Ptmp;
    SignificantBetasDual(isnan(SignificantBetasDual)==1)=0;
    %Do not allow negative power amplitudes and set all negative amplitudes
    %to almost zero
    SignificantBetasDual(SignificantBetasDual<0) =0.0001;

    %***************RERUN GLM WITH REFINED SPECTRA*************************
    disp(['Starting second step: Rerunning GLM with refined spectra for ' num2str(Iter) ' iterations'])
    XRefined = SignificantBetasDual;
    if (Iter ==1) && (FitRespBool == 1 || FitCardBool == 1) 
        BetasRefined = zeros(Num_RO,Num_PE,Slices,2);
        PValsRefined = ones(Num_RO,Num_PE,Slices,2);
    else
        BetasRefined = zeros(Num_RO,Num_PE,Slices,3);
        PValsRefined = ones(Num_RO,Num_PE,Slices,3);
    end
    clear YRefined
    parfor i = 1:Num_RO
        if(i==Num_RO/2)
            disp('Half way through refined GLM.');
        elseif (i==Num_RO)
            disp('Refined GLM finished.');
        end
        for j = 1:Num_PE
            for k = 1:Slices
                if any(Resids(i,j,k,:))
                    YRefined = squeeze(AllVoxelSpectraGLM(i,j,k,:));
                    [Betastmp, devtmp, stats] = glmfit(XRefined,YRefined,'normal','constant','off');
                    BetasRefined(i,j,k,:) = Betastmp;
                    PValsRefined(i,j,k,:) = stats.p;
                end
            end
        end
    end

    %Set up refined spatial maps
    Ptmp = PValsRefined; Ptmp(Ptmp<pVal) = 2; Ptmp(Ptmp<2) = 0; Ptmp(Ptmp>0) = 1;
    SBetasRefined = BetasRefined.*Ptmp;
    SBetasRefined(isnan(SBetasRefined)==1)=0;

    %Do not allow negative power amplitudes and set all negative amplitudes
    %to zero
    SBetasRefined(SBetasRefined<0) =0;
    
    
    %Save normalised refined spectra
    if FitRespBool == 1 && FitCardBool == 0 && Iter ==1 
        NewCardSpec     = SignificantBetasDual(:,2)./sum(SignificantBetasDual(:,2));
        %Assign base line to respiratory spectrum
        NewRespSpec     = SignificantBetasDual(:,1)-1; NewRespSpec(NewRespSpec<=0) = min(NewCardSpec);%set negative values to noise base line in external recording
        NewRespSpec     = NewRespSpec./sum(NewRespSpec);
        
        %Assign
        RefinedCardSpec = NewCardSpec;
        RefinedRespSpec = NewRespSpec; 
        
    elseif FitCardBool == 1 && FitRespBool == 0 && Iter ==1 
        NewRespSpec     = SignificantBetasDual(:,2)./sum(SignificantBetasDual(:,2));
        %Assign base line to respiratory spectrum
        NewCardSpec = SignificantBetasDual(:,1)-1; NewCardSpec(NewCardSpec<=0) = min(NewRespSpec); %set negative values to noise base line in external recording
        NewCardSpec = NewCardSpec./sum(NewCardSpec);
        
        %assign
        RefinedCardSpec = NewCardSpec;
        RefinedRespSpec = NewRespSpec;
        
    elseif FitCardBool == 1 && FitRespBool == 1 && Iter ==1 
        %assign lower frequencies to respiration
        NewRespSpec     = SignificantBetasDual(:,1)- 1; NewRespSpec(end/3:end) = min(abs(NewRespSpec));
        %assign higher frequencies to cardiac
        NewCardSpec     = SignificantBetasDual(:,1)- 1; NewCardSpec(1:end/3) = min(abs(NewCardSpec)); 
       
        %assign
        RefinedCardSpec = NewCardSpec;
        RefinedRespSpec = NewRespSpec;
    else
        NewBL       = SignificantBetasDual(:,1);
        NewCardSpec = SignificantBetasDual(:,2);NewCardSpec(NewCardSpec<=0) = 0.0001;
        NewCardSpec = NewCardSpec./sum(NewCardSpec);
        NewRespSpec = SignificantBetasDual(:,3);NewRespSpec(NewRespSpec<=0) = 0.0001;
        NewRespSpec = NewRespSpec./sum(NewRespSpec);

        RefinedBL = NewBL;
        RefinedCardSpec = NewCardSpec;
        RefinedRespSpec = NewRespSpec;
    end
    %plot
        plot(f(fminGLMindex+1:end),RefinedCardSpec,'Color', [0  0.5 0]);hold on
        plot(f(fminGLMindex+1:end),RefinedRespSpec','Color', [0  0  0.5]);
        legend('Cardiac refined', 'Respiratory refined');
        title (['Iteration: ' num2str(Iter) ]);
        drawnow
    %Calculate convergence level
    ConvergedResp =  sum(abs(SpecRespGLM - NewRespSpec'));
    ConvergedCard =  sum(abs(SpecCardGLM - NewCardSpec'));
    
    %Increase iteration counter
    Iter = Iter +1;
    disp(['Convergence cardiac     ' num2str(ConvergedCard) ]);
    disp(['Convergence respiratory ' num2str(ConvergedResp) ]);
    if ConvergedResp < ConvLevel && ConvergedCard < ConvLevel
        ConvBool = 1;
    end
    
end
RefinedSpec = [RefinedBL, RefinedCardSpec, RefinedRespSpec];
toc
delete(gcp('nocreate'))
end