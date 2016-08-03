    clear; clc; close all;

    trainImage = rgb2gray(imread('TrainingMats/catinhat.jpg'));
    testImage = rgb2gray(imread('testmat.jpg'));

    trainPoints = detectSURFFeatures(trainImage);
    testPoints = detectSURFFeatures(testImage);

    figure;
    imshow(trainImage);
    title('100 Strongest Features');
    hold on;
    plot(selectStrongest(trainPoints, 100));

    figure;
    imshow(testImage);
    title('3000 Strongest Features');
    hold on;
    plot(selectStrongest(testPoints, 3000));
    % export_fig print.png -native

    [trainFeatures, trainPoints] = extractFeatures(trainImage, trainPoints);
    [testFeatures, testPoints] = extractFeatures(testImage, testPoints);

    imgPairs = matchFeatures(trainFeatures, testFeatures);
    matchedTrainingPoints = trainPoints(imgPairs(:, 1), :);
    matchedTestingPoints = testPoints(imgPairs(:, 2), :);

    [tform, inlierTrainPoints, inlierTestPoints] = estimateGeometricTransform(matchedTrainingPoints, matchedTestingPoints, 'affine');

    figure;
    showMatchedFeatures(trainImage, testImage, inlierTrainPoints, inlierTestPoints, 'montage');
    title('Matching Points');

    boxPolygon = [1, 1; size(trainImage, 2), 1; size(trainImage, 2), size(trainImage, 1); 1, size(trainImage, 1); 1, 1];
    newBoxPolygon = transformPointsForward(tform, boxPolygon);

    figure;
    imshow(testImage);
    hold on;
    line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'green', 'LineWidth', 1);
    title('Bounding Box');