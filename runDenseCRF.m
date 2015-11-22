% perform dense CRF for usefulness study

function runDenseCRF(fold)

run('configParas');

% textonboost path
% TODO: need to change the name of the texonboost file
textonpath = sprintf('%s/output_f%d/split_%d/evaluation/', textonboostDir, fold, fold); 

% testing files
filesID = fopen(sprintf('%s/dense/split_%d.txt', gtPath, fold), 'r');
testingFiles = fscanf(filesID,'%d_%d', [2 Inf]);
fclose(filesID);

%result dir
resultDir = '../denseCRF/';

dim = [1408, 376, 14];
numlabels = dim(3);
n = 0;

for i = 1:length(testingFiles(1, :))
    
    seq = testingFiles(1, i);
    currFrame = testingFiles(2, i);
    
    % color path
    folderImg = sprintf('%s/2013_05_28_drive_%04d_sync/', baseDir, seq);
    img = imread(sprintf('%simage_0%d/data_rect/%010d.png', folderImg, ... 
            0, currFrame));
        
     % textonboost
    [textonboost, L_t] = loadTextonboost(textonpath, numlabels, currFrame, 0);
        unary{1} = single(-1*textonboost');
    
    %gt
    gtDir = sprintf('%s/seq_%04d/binary_semantic/', gtPath, seq);
    
    % do the hard work
  
    tmp = -1*textonboost;
    tmp = reshape(tmp, size(img, 1), size(img, 2), numLabels);
    % convert to cpp index
    tmp = permute(tmp, [2 1 3]);
    tmp = reshape(tmp, size(img, 1)*size(img, 2), numLabels);

    u = tmp'*fullcrfPara.uw;

    tmpImg = reshape(img, [], 3);
    tmpImg = tmpImg';
    tmpImg = reshape(tmpImg, 3, size(img, 1), size(img, 2));
    tmpImg = permute(tmpImg, [1 3 2]);

    [L, prob] = fullCRFinfer(single(u), uint8(tmpImg), fullcrfPara.s, fullcrfPara.s, ...  
        fullcrfPara.sw, fullcrfPara.bl, fullcrfPara.bl, fullcrfPara.bc, fullcrfPara.bc, ... 
        fullcrfPara.bc, fullcrfPara.bw, size(img, 2), size(img, 1), numLabels);

    map = (reshape(L, size(img, 2), size(img, 1)))';

    [gt n] = loadGT(currFrame, 0, n, gtDir);
    if (isempty(gt)) continue; end
    
    if (~exist([resultDir '/result'], 'dir'))
        mkdir([resultDir '/result']);
    end

    [intersection0, union0, acc0, classweight0, totalValid0, ~] = ... 
        parseAcc(gt, map+1, numlabels, img);
    
    fprintf('Processing %010d, acc = %f. \n', currFrame, acc0);
    
    % visual result
    inferlabels.label = map;

    % acc
    inferlabels.acc = acc0;

    % ji
    %inferlabels.ji{method} = jacIdx0;
    inferlabels.intersect = intersection0;
    inferlabels.union = union0;

    inferlabels.weight = classweight0;
    inferlabels.totalValid = totalValid0;

    % frame number
    inferlabels.frame =  currFrame;
    inferlabels.seq =  seq;

    save(sprintf('%sresult/result_%04d_%06d.mat', resultDir, seq, currFrame), 'inferlabels');
     
end
