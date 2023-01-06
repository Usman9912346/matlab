% Settings ================================================================
reclassify = false;
% directory of folder where the reports are
folderDir = pwd + "/Reports"; % default location of all reports
%folderDir = "Reports by Publisher/CSB/"; % substitute for CSB reports only, more accurate results
%folderDir = "Reports by Publisher/NTSB/"; % substitute for NTSB reports only
getImpSection = false;
limitResultToOneAndZero = true;
% End of Settings =========================================================
classTable = dataCompiler(folderDir,getImpSection); 

if reclassify == true
    classGroup = [4 3 5 2 8 3 2 1 3 2 2 6 2 6 4];
    newClassGroup = [1];
    currentColNum = 1;
    for i = 1:numel(classGroup)
        currentColNum = currentColNum + classGroup(i);
        newClassGroup = [newClassGroup currentColNum];
    end
 
    classMat = cell2mat(classTable(:,4:end));
    reclassCell = [];
    for i = 1:size(classMat,1)
        currentRow = [];
        for j = 2:numel(newClassGroup)
            currentTotal = sum(classMat(i,newClassGroup(j - 1) + 1:newClassGroup(j) - 1));
            currentRow = [currentRow currentTotal];
        end
        reclassCell = [reclassCell;currentRow];
    end
        if limitResultToOneAndZero == true
        reclassCell(reclassCell > 1) = 1;
    end
else
    reclassCell = classTable(:,4:end);
end

debug = []; 
mdlArray = {}; 
bags = {};
if ~iscell(reclassCell)
    reclassCell = num2cell(reclassCell);
end
newClassTable = [classTable(:,1:3) reclassCell]; 
    
    cvp = cvpartition(cell2mat(newClassTable(:,4,:)),'Holdout',0.1);
    dataTrain = newClassTable(cvp.training,:);
    dataTest = newClassTable(cvp.test,:);

    textDataTrain = [dataTrain{:,3}]';
    textDataTest = [dataTest{:,3}]';
    documents = preprocess(textDataTrain);
    
    bag = bagOfWords(documents);
    bag = removeInfrequentWords(bag,2);
    [bag,idx] = removeEmptyDocuments(bag);
    XTrain = bag.Counts;
    
    bags{end+1,1} = bag;

    documentsTest = preprocess(textDataTest);
    XTest = encode(bag,documentsTest);
    YTrain = cell2mat(dataTrain(:,4:end));
    YTest = cell2mat(dataTest(:,4:end));
    YTrain(idx,:) = [];

for i = 1:size(YTrain,2)  
    mdl = fitcecoc(XTrain,YTrain(:,i),'Learners','linear');
    mdlArray{end + 1,1} = mdl;
    YPred = predict(mdl,XTest);
    YPreds(:,i)=YPred;
    
end

[parentdir,~,~]=fileparts(pwd);
parentdir = pwd + "/Uploads";
uploads = dir(fullfile(parentdir,"*.pdf"));
temp = uploads.name;
temp = parentdir + "/" + temp;
filepath = temp;
temp = strsplit(extractFileText(temp),"\n");

size_tmp = size(temp);
N_rows_file = size_tmp(2);

str=strjoin(temp);
documentsNew = preprocess(str);

for j = 1:size(mdlArray,1)
    XNew = encode(bags{1,1},documentsNew);
    labelsNew = predict(mdlArray{j},XNew);
    result1(j,:) = labelsNew;
end

[Out, Tdat] = print_string(result1);
OutString = string(Out(2:end,1));
OutString = strrep(OutString,'_',' ');

[Accuracy,Precision,Recall,F1Score]=accmetrics(YPreds,YTest);
AccuracyString={};
AccuracyString = {'Accuracy';Accuracy;'Precision';Precision;'Recall';Recall;'F1Score';F1Score}
AccuracyString=string(AccuracyString);

[parentdir,~,~]=fileparts(pwd);
 savepath = pwd;
addpath(savepath);

fid = fopen([savepath + "/Results/" + 'factors.txt'],'wt');
fprintf(fid, '%s\n', OutString);
fclose(fid);

fid2 = fopen([savepath + "/Results/" + 'accuracies.txt'],'wt');
fprintf(fid2, '%s\n', AccuracyString);
fclose(fid2);
