% This segmentation implementation is reference from: 
% [1] “Semantic Segmentation With Deep Learning,” Semantic segmentation with Deep Learning - MATLAB &amp; Simulink - MathWorks United Kingdom, https://uk.mathworks.com/help/vision/ug/semantic-segmentation-with-deep-learning.html (accessed May 15, 2023). 

% This segmentation model is built from scratch with U-net architecture

% Clean everything and load the image and label file path
close all;
clc;
clear;

% Read the folder path and get the file path
% Please modify the file path to the location where the image folder located at
% if want to run the code.
dataDir = "/Users/User/Desktop/computer_vision/COMP3007_20196637/daffodilSeg/";

% load the image and pixel label from the directory
imageDir = fullfile(dataDir,'ImagesRsz256');
pixelDir = fullfile(dataDir,'LabelsRsz256');

%% Set the image and pixel directory
% Store image directory
imageDS = imageDatastore(imageDir);

% set class names to match the values in the label images
classNames = ["flower" "background"];

% set the label for corresponding class
% '1' = flower, '3' = background
pixelLabelID = [1 3];

% Create a pixel label store that store all the pixel labels and names
pixelDS = pixelLabelDatastore(pixelDir,classNames,pixelLabelID);


%% Split and preprocessing dataset
% Split the data into train, validation and test set
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = splitDataset(imageDS,pixelDS);

%% Resize image and apply data augmentation
% Resize image to fix the size to [256 256]
image_size = [256 256];

% Parameter settings for image augmentation 
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandXScale',[0.8 0.8], ...
    'RandYScale',[0.8 0.8], ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);

% Combine the training image and pixel label by apply data augmentation and
% resize function
trainingData = pixelLabelImageDatastore(imdsTrain, pxdsTrain,'DataAugmentation',imageAugmenter,'OutputSize',image_size)

% Combine validation image and label for validation during training
dsVal = combine(imdsVal,pxdsVal)

%%  Create CNN network with input layer, down-sampling, up-sampling and fully connected layer
% Set the input layer size for CNN network
inputSize = [256 256 3];
imgLayer = imageInputLayer(inputSize)


% Down-sampling network
% Set the value for the down-sampling network
filterSize = 3;
numFilters = 64;
conv = convolution2dLayer(filterSize,numFilters,'Padding',1);
relu = reluLayer();

% Set the max pooling size
poolSize = 2;
maxPoolDownsample = maxPooling2dLayer(poolSize,'Stride',2);

% Combine the convolutional layer, relu layer and maxpooling layer together
downsamplingLayers = [
    conv
    relu
    maxPoolDownsample
    conv
    relu
    maxPoolDownsample
    ]

% Up-sampling network
% Set the value for the Up-sampling network
filterSize = 4;

% This function consist of upsampling and convolution layer
transposedConvUpsample = transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping',1);

% Combine the transposed convolutional layer, relu layer and maxpooling layer together
upsamplingLayers = [
    transposedConvUpsample
    relu
    transposedConvUpsample
    relu
    ]

% Pixel classification layer
% match the number of output labels
numClasses = 2;
class_conv = convolution2dLayer(1,numClasses);

finalLayers = [
    class_conv
    softmaxLayer()
    pixelClassificationLayer()
    ]

% Stack the input layer, down-sampling, up-sampling and classification
% layer together
net = [
    imgLayer    
    downsamplingLayers
    upsamplingLayers
    finalLayers
    ]

%% Set the hyperparameter settings and train the model
% Model hyperparameter settings
opts = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',50, ... % how long to train for
    'L2Regularization', 1e-4, ...
    'MiniBatchSize',64, ...
    'ValidationData',dsVal, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',10);


% Train the network
net = trainNetwork(trainingData,net,opts);

%% Save the trained model
save('segmentnet.mat', 'net') %filename, variable

%% Create out folder to store the result produced when test with testing dataset

% Create 'out' folder if not exist
if ~exist('out', 'dir')
   mkdir('out')
end

% Perform segmentation, save output images 'out' folder
% If there is not 'out' folder, please create the out folder manually
pxdsResults = semanticseg(imdsTest,net,"WriteLocation","out");

%% Evaluate the result
% show the first 10 image with predicted segmentation
for i = 1:10
    overlayOut = labeloverlay(readimage(imdsTest,i),readimage(pxdsResults,i)); %overlay
    figure
    imshow(overlayOut);
    title('overlayOut ' + i)
end

% calculate metrics for segmentation result
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest);
figure;

% Show the accuracy and iou of the results
metrics.ClassMetrics

% display confusion matrix of predicting result
cm = confusionchart(metrics.ConfusionMatrix.Variables,classNames,Normalization='row-normalized');
cm.Title = 'Segmentation for flower and background'
figure;

% Show the mean IOU of the result
IOU = metrics.ImageMetrics.MeanIoU

% plot histogram of IOU values
h = histogram(IOU)

%% Split the dataset into train, valid and test set

% This function is reference from the matlab official site which used to
% split the images and pixel label into training, validation and testing
% set with corresponding number.

% Reference: 
% [1] “Semantic Segmentation Using Deep Learning,” MATLAB &amp; Simulink - MathWorks United Kingdom, https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html (accessed May 16, 2023). 

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = splitDataset(imageDS,pixelDS)
    % Partition CamVid data by randomly selecting 60% of the data for training. The
    % rest is used for testing.
        
    % Set initial random state for example reproducibility.
    rng(0); 
    numFiles = numel(imageDS.Files);
    shuffledIndices = randperm(numFiles);

    % Use 60% of the images for training.
    numTrain = round(0.60 * numFiles);
    trainingIdx = shuffledIndices(1:numTrain);

    % Use 20% of the images for validation
    numVal = round(0.20 * numFiles);
    valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

    % Use the rest for testing.
    testIdx = shuffledIndices(numTrain+numVal+1:end);

    % Create image datastores for training and test.
    trainingImages = imageDS.Files(trainingIdx);
    valImages = imageDS.Files(valIdx);
    testImages = imageDS.Files(testIdx);
    imdsTrain = imageDatastore(trainingImages);
    imdsVal = imageDatastore(valImages);
    imdsTest = imageDatastore(testImages);

    % Extract class and label IDs info.
    classes = pixelDS.ClassNames;
    labelIDs = [1 3];
    
    % Create pixel label datastores for training and test.
    trainingLabels = pixelDS.Files(trainingIdx);
    valLabels = pixelDS.Files(valIdx);
    testLabels = pixelDS.Files(testIdx);
    pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
    pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
    pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end