% This classification implementation is reference from: 
% [1] "Train Deep Learning Network to Classify New Images," MATLAB &amp; Simulink - MathWorks United Kingdom, https://uk.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html#TransferLearningUsingGoogLeNetExample-3 (accessed May 15, 2023). 

% This classification model is using transfer learning on ResNet-50 model


% Clean everything and load the image and label file path
clc;
close all;
clear;

%%
% % Remind that the 17flowers dataset used are organized into 17
% % folder with corresponding label. Please organize the dataset before run
% % the code

% % This part is help to organize the image files store in the 17flowers
% % folder

% srcroot = "17flowers/";
% srcdir = dir(srcroot);
% srcfolder = {srcdir.name};
% srcfolder = srcfolder(~ismember(srcfolder, {'.', '..'}));
% 
% flower_class = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus", "iris", "tigerlily", "tulip", "fritillary", "sunflower", "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup", "windflower", "pansy"];
% x = 1;
% for i = 1:length(flower_class)
%     folder_name = flower_class{i};
%     mkdir("17flowers/" + flower_class{i});
%     files = dir(sprintf('*.jpg', folder_name));
%     for j = x: x + 79
%         filename = "17flowers/" + srcfolder{j};
%         disp(filename)
%           
%         destroot = "17flowers/" + flower_class{i};
%         movefile(filename, destroot);
%     end
%     x = x + 80;
% end

%%
% Read the folder path and get the file path
% Please modify the file path to the location where the image folder located at
% if want to run the code.
dataDir = "/Users/User/Desktop/computer_vision/COMP3007_20196637/";
imDir = fullfile(dataDir,'17flowers');
imds = imageDatastore(imDir, 'LabelSource','foldernames','IncludeSubfolders',true)


%% Split the dataset to 60:20:20 ratio randomly

[trainingSet, validSet, testSet] = splitEachLabel(imds, 0.6, 0.2,'randomized')
%% Load te pretrained network

net = resnet50();
% analyzeNetwork(net)

%% Perform Data Augmentation on training dataset to increase the variation 

% Obtain the size of the input layer from model to resize the training
% dataset
imageSize = net.Layers(1).InputSize;

% Parameter settings for data augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandXScale',[0.8 0.8], ...
    'RandYScale',[0.8 0.8], ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);

% Apply the data augmentation and resize the image on training dataset
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet,'DataAugmentation',imageAugmenter);

% Resize the validation dataset
augmentedValidSet = augmentedImageDatastore(imageSize, validSet);

%% Find the last layer to replace to match the number of output classess (17)
lgraph = layerGraph(net)
% analyzeNetwork(lgraph)
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

%% Replace the last layer
numClasses = numel(categories(trainingSet.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','17Flowers_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','17Flowers_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

% Replace the learnable layer
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% Replace the classification layer
newClassLayer = classificationLayer('Name','17Flowers_class');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% analyzeNetwork(lgraph)
%% Set the hyperparameter and train the network

% Hyperparameter settings to train the CNN model
options = trainingOptions('adam', ...
    'MiniBatchSize',64, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedValidSet, ...
    'L2Regularization', 1e-4, ...
    'MaxEpochs',30)

% Train the network
net = trainNetwork(augmentedTrainingSet, lgraph, options)

%% Save the model trained
save('classnet.mat', 'net') %filename, variable

%% evaluate the performance on testing dataset
augmentedTestSet = augmentedImageDatastore(imageSize, testSet)
[YPred, probs] = classify(net, augmentedTestSet);
accuracy = mean(YPred == testSet.Labels)

% Calculate precision, recall and f1-score
[cm, order] = confusionmat(testSet.Labels, YPred);
cmt = cm'
diagonal = diag(cmt) 
sum_of_rows = sum(cmt, 2) 

precision = diagonal ./ sum_of_rows
overall_precision = mean(precision) 
sum_of_columns = sum(cmt, 1) 
recall = diagonal ./ sum_of_columns' 
overall_recall = mean(recall) 
f1_score = 2*((overall_precision*overall_recall)/(overall_precision+overall_recall))

%% Evlaute the certaion result with predicted label
idx = [1 2 3 4 5 6];
figure
for i = 1:numel(idx)
    subplot(3,3,i)
    I = readimage(testSet,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(label)
end

%% Function to find the last layer to replace

% This function is pre-built function provided by MATLAB to help and search
% for the learnable layer and classification layer in the pre-built network

% You can find it by enter 'ls findLayersToReplace.m' in command window

function [learnableLayer,classLayer] = findLayersToReplace(lgraph)
    % findLayersToReplace(lgraph) finds the single classification layer and the
    % preceding learnable (fully connected or convolutional) layer of the layer
    % graph lgraph.
    
    % Copyright 2021 The MathWorks, Inc.
    
    if ~isa(lgraph,'nnet.cnn.LayerGraph')
        error('Argument must be a LayerGraph object.')
    end
    
    % Get source, destination, and layer names.
    src = string(lgraph.Connections.Source);
    dst = string(lgraph.Connections.Destination);
    layerNames = string({lgraph.Layers.Name}');
    
    % Find the classification layer. The layer graph must have a single
    % classification layer.
    isClassificationLayer = arrayfun(@(l) ...
        (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|isa(l,'nnet.layer.ClassificationLayer')), ...
        lgraph.Layers);

    if sum(isClassificationLayer) ~= 1
        error('Layer graph must have a single classification layer.')
    end
    classLayer = lgraph.Layers(isClassificationLayer);
    
    
    % Traverse the layer graph in reverse starting from the classification
    % layer. If the network branches, throw an error.
    currentLayerIdx = find(isClassificationLayer);
    while true
        
        if numel(currentLayerIdx) ~= 1
            error('Layer graph must have a single learnable layer preceding the classification layer.')
        end
        
        currentLayerType = class(lgraph.Layers(currentLayerIdx));
        isLearnableLayer = ismember(currentLayerType, ...
            ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);
        
        if isLearnableLayer
            learnableLayer =  lgraph.Layers(currentLayerIdx);
            return
        end
        
        currentDstIdx = find(layerNames(currentLayerIdx) == dst);
        currentLayerIdx = find(src(currentDstIdx) == layerNames); %#ok<FNDSB>
        
    end

end
