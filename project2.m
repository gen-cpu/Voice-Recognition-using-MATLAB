% --- Initial Settings ---
clear; clc; close all;
rng(42);  % Set random seed for reproducibility

% --- Define Main Project Folder ---
projectFolder = '.';

% --- Collect Speaker Folders ---
subFolders = dir(projectFolder);
speakers = {subFolders([subFolders.isdir] & ~startsWith({subFolders.name}, '.')).name};
if isempty(speakers)
    error('No speaker folders found.');
end

% --- Convert .waptt files to .wav using FFmpeg ---
for speakerIdx = 1:length(speakers)
    speakerPath = fullfile(projectFolder, speakers{speakerIdx});
    emotionFolders = dir(speakerPath);
    emotionFolders = emotionFolders([emotionFolders.isdir] & ~startsWith({emotionFolders.name}, '.'));
    
    for e = 1:length(emotionFolders)
        emotionPath = fullfile(speakerPath, emotionFolders(e).name);
        wapttFiles = dir(fullfile(emotionPath, '*.waptt'));
        
        fprintf('Processing: %s/%s | Number of files: %d\n', ...
            speakers{speakerIdx}, emotionFolders(e).name, length(wapttFiles));
        
        for f = 1:length(wapttFiles)
            originalFile = fullfile(emotionPath, wapttFiles(f).name);
            [~, name, ~] = fileparts(wapttFiles(f).name);
            opusFile = fullfile(emotionPath, [name '.opus']);
            wavFile = fullfile(emotionPath, [name '.wav']);
            
            try
                movefile(originalFile, opusFile);  % Rename .waptt to .opus
                ffmpegCommand = sprintf('ffmpeg -y -i "%s" -ar 16000 "%s"', opusFile, wavFile);
                [status, cmdout] = system(ffmpegCommand);
                
                if status == 0 && exist(wavFile, 'file')
                    delete(opusFile);  % Delete intermediate .opus file after successful conversion
                else
                    error('Conversion failed: %s', cmdout);
                end
            catch
                warning('Failed to process file: %s', wapttFiles(f).name);
                continue;
            end
        end
    end
end
% --- Extract emotion categories from first speaker folder ---
exampleSpeakerPath = fullfile(projectFolder, speakers{1});
exampleEmotions = dir(exampleSpeakerPath);
emotions = {exampleEmotions([exampleEmotions.isdir]).name}';
emotions = emotions(~startsWith(emotions, '.'));  % Remove system folders like "."
numEmotions = length(emotions);

% --- Initialize Variables for Feature Storage ---
fixedFs = 16000;  % Standard sampling rate for all audio
speakerFeatures = [];
speakerLabels = [];
emotionFeatures = [];
emotionLabels = [];

% --- Extract Features from All Audio Files ---
for speakerIdx = 1:length(speakers)
    speakerPath = fullfile(projectFolder, speakers{speakerIdx});
    emotionFolders = dir(speakerPath);
    emotionFolders = emotionFolders([emotionFolders.isdir] & ~startsWith({emotionFolders.name}, '.'));
    
    for emotionIdx = 1:length(emotionFolders)
        emotionPath = fullfile(speakerPath, emotionFolders(emotionIdx).name);
        samples = dir(fullfile(emotionPath, '*.wav'));
        if isempty(samples), continue; end  % Skip if no audio files
        
        for sample = 1:length(samples)
            filename = fullfile(emotionPath, samples(sample).name);
            [audio, originalFs] = audioread(filename);
            audio = resample(audio, fixedFs, originalFs);  % Resample to fixedFs

            % Apply standard pre-processing
            processed = bandpass(audio, [300, 3400], fixedFs);
            processed = processed / max(abs(processed));
            processed = sign(processed) .* sqrt(abs(processed));

            % Extract MFCC and related features
            mfccs = mfcc(processed, fixedFs, 'NumCoeffs', 14);
            delta = [zeros(1, size(mfccs, 2)); diff(mfccs)];
            deltaDelta = [zeros(1, size(delta, 2)); diff(delta)];
            combined = [mfccs, delta, deltaDelta];
            meanFeat = mean(combined, 1);
            pitchFeat = mean(pitch(processed, fixedFs));
            energy = rms(processed);
            features = [meanFeat, pitchFeat, energy];

            % Duplicate 3 times for balancing
            for i = 1:3
                speakerFeatures = [speakerFeatures; features];
                speakerLabels = [speakerLabels; speakerIdx];
                emotionFeatures = [emotionFeatures; features];
                emotionLabels = [emotionLabels; emotionIdx];
            end

            % --- Data Augmentation Section ---
            audioVariants = {
                audio + 0.005 * randn(size(audio));                          % Noise Injection
                resample(audio, round(fixedFs * 1.1), fixedFs);              % Time Stretching
                audio .* sin(2 * pi * (1:length(audio))' / 2000);           % Pitch Shift
                circshift(audio, 800);                                       % Time Shifting
                audio * 1.25;                                                % Volume Scaling
                conv(audio, [1 zeros(1, 100) 0.6 zeros(1, 200) 0.3], 'same');% Reverb
                highpass(audio, 500, fixedFs);                               % Equalization
                min(max(audio, -0.3), 0.3);                                  % Clipping
            };

            % Extract features from augmented audio
            for aug = 1:length(audioVariants)
                a = audioVariants{aug};
                a = bandpass(a, [300, 3400], fixedFs);
                a = a / max(abs(a));
                a = sign(a) .* sqrt(abs(a));
                
                mfccs = mfcc(a, fixedFs, 'NumCoeffs', 14);
                delta = [zeros(1, size(mfccs, 2)); diff(mfccs)];
                deltaDelta = [zeros(1, size(delta, 2)); diff(delta)];
                combined = [mfccs, delta, deltaDelta];
                meanFeat = mean(combined, 1);
                pitchFeat = mean(pitch(a, fixedFs));
                energy = rms(a);
                features = [meanFeat, pitchFeat, energy];

                speakerFeatures = [speakerFeatures; features];
                speakerLabels = [speakerLabels; speakerIdx];
                emotionFeatures = [emotionFeatures; features];
                emotionLabels = [emotionLabels; emotionIdx];
            end
        end
    end
end
% --- Feature Selection using fscmrmr (top 20 features) ---
disp('Selecting top features using fscmrmr...');
speakerIdxKeep = fscmrmr(speakerFeatures, speakerLabels);
speakerTopIdx = speakerIdxKeep(1:min(20, length(speakerIdxKeep)));

emotionIdxKeep = fscmrmr(emotionFeatures, emotionLabels);
emotionTopIdx = emotionIdxKeep(1:min(20, length(emotionIdxKeep)));

% Reduce the features to selected ones
speakerFeatures = speakerFeatures(:, speakerTopIdx);
emotionFeatures = emotionFeatures(:, emotionTopIdx);

% Save selected feature indices for live prediction
save('selectedFeaturesIdx.mat', 'speakerTopIdx', 'emotionTopIdx');

% --- Split Data for Training and Testing (Stratified HoldOut) ---
cvSpeaker = cvpartition(speakerLabels, 'HoldOut', 0.2, 'Stratify', true);
trainingIdx = training(cvSpeaker);
testIdx = test(cvSpeaker);



% --- Train Final Models using Best K ---
speakerModel = fitcknn(speakerFeatures(trainingIdx, :), speakerLabels(trainingIdx), ...
    'NumNeighbors', 3, 'Standardize', true);
emotionModel = fitcknn(emotionFeatures(trainingIdx, :), emotionLabels(trainingIdx), ...
    'NumNeighbors', 3, 'Standardize', true);

% --- Evaluate Models ---
speakerPreds = predict(speakerModel, speakerFeatures(testIdx, :));
emotionPreds = predict(emotionModel, emotionFeatures(testIdx, :));

% --- Confusion Matrices and Accuracy ---
cmSpeaker = confusionmat(speakerLabels(testIdx), speakerPreds);
cmEmotion = confusionmat(emotionLabels(testIdx), emotionPreds);

accuracySpeaker = sum(diag(cmSpeaker)) / sum(cmSpeaker(:));
accuracyEmotion = sum(diag(cmEmotion)) / sum(cmEmotion(:));

disp('---------------------------------------------');
disp('Model Evaluation Summary:');
fprintf('- Speaker Recognition Accuracy: %.2f%%\n', accuracySpeaker * 100);
fprintf('- Emotion Recognition Accuracy: %.2f%%\n', accuracyEmotion * 100);
disp('---------------------------------------------');

% --- Visualization: Confusion Matrix (Speaker) ---
figure;
confusionchart(cmSpeaker, speakers, ...
    'Title', 'Confusion Matrix - Speaker Recognition', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
set(gcf, 'Name', 'Speaker Confusion Matrix');

% --- Visualization: Confusion Matrix (Emotion) ---
figure;
confusionchart(cmEmotion, emotions, ...
    'Title', 'Confusion Matrix - Emotion Recognition', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
set(gcf, 'Name', 'Emotion Confusion Matrix');


% --- Start interactive audio testing loop ---
disp('------------------------------------------');
disp('Interactive speaker & emotion recognition test');
disp('Type "exit" to quit.');

% Initialize audio recorder (even if not used in this version)
recorder = audiorecorder(fixedFs, 16, 1);
isRunning = true;

while isRunning
    disp('Press Enter to select a WAV file or type "exit"...');
    userInput = input('', 's');

    % Exit condition
    if strcmpi(userInput, 'exit')
        isRunning = false;
        continue;
    end

    % Select audio file
    [fileName, filePath] = uigetfile({'*.wav'}, 'Select a WAV file (1–10 seconds)');
    if isequal(fileName, 0)
        disp('No file selected.');
        continue;
    end
    fullFilePath = fullfile(filePath, fileName);
    [audioLive, fsLive] = audioread(fullFilePath);

    % Check duration
    durationSec = length(audioLive) / fsLive;
    if durationSec < 1
        error('Audio too short. Must be at least 1 second (%.2f sec)', durationSec);
    elseif durationSec > 10
        audioLive = audioLive(1 : 10 * fsLive);  % Trim to first 10 seconds
    end

    % Resample to 16kHz
    audioLive = resample(audioLive, fixedFs, fsLive);

    % Preprocessing: bandpass, normalization, compression
    audioLive = bandpass(audioLive, [300, 3400], fixedFs);
    audioLive = audioLive / max(abs(audioLive));
    audioLive = sign(audioLive) .* sqrt(abs(audioLive));

    % Feature extraction: MFCC, delta, delta-delta, pitch, energy
    mfccsLive = mfcc(audioLive, fixedFs, 'NumCoeffs', 14);
    deltaLive = [zeros(1, size(mfccsLive, 2)); diff(mfccsLive, 1, 1)];
    deltaDeltaLive = [zeros(1, size(deltaLive, 2)); diff(deltaLive, 1, 1)];
    combinedFeaturesLive = [mfccsLive, deltaLive, deltaDeltaLive];
    meanFeaturesLive = mean(combinedFeaturesLive, 1);
    pitchLive = mean(pitch(audioLive, fixedFs));
    energyLive = rms(audioLive);
    fullLiveFeatures = [meanFeaturesLive, pitchLive, energyLive];

    % Load selected feature indices from training
    load('selectedFeaturesIdx.mat', 'speakerTopIdx', 'emotionTopIdx');
    liveFeatures = fullLiveFeatures(speakerTopIdx);
    liveEmotionFeatures = fullLiveFeatures(emotionTopIdx);

    % Predict speaker and emotion using KNN
    [speakerLabel, speakerScores, speakerDistances] = predict(speakerModel, liveFeatures);
    [emotionLabel, emotionScores] = predict(emotionModel, liveEmotionFeatures);

    % Extract confidence and distance
    speakerConf = speakerScores(speakerLabel);
    minSpeakerDist = min(speakerDistances);
    emotionConf = emotionScores(emotionLabel);

    % Thresholds for speaker and emotion decision
    speakerThreshold = 0.7;
    speakerDistThreshold = 1.5;
    emotionThreshold = 0.7;

    % Speaker result
    if speakerConf >= speakerThreshold && minSpeakerDist <= speakerDistThreshold
        fprintf('Speaker: %s | Confidence: %.2f | Distance: %.2f\n', ...
                speakers{speakerLabel}, speakerConf, minSpeakerDist);
    else
        fprintf('Unknown speaker (Confidence: %.2f | Distance: %.2f)\n', ...
                speakerConf, minSpeakerDist);
    end

    % Emotion result
    if emotionConf >= emotionThreshold
        fprintf('Emotion: %s | Confidence: %.2f\n', ...
                emotions{emotionLabel}, emotionConf);
    else
        fprintf('Unclear emotion (Confidence: %.2f)\n', emotionConf);
    end
end
