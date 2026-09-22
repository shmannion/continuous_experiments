%import Polhemus csv
%import behavioral csv


% column 2: sample No
% column 3: trigger for polhemus
% column 4,5 6:  X, Y ,Z coordinate respectively.

clear all

% datapath='C:\Users\kyvko\Dropbox\DTU Copenhagen\Experiments\SecondMirror\Data\polhemus\Exp\'
datapath='your path with polhemus data csv'

% datapath='D:\KK\Polhemus\'

f=dir(fullfile(datapath,'*.csv'))



%% you dont need this part if you just load the raw data

for k=1:length(f)

    Polhemus_temp = readtable([datapath f(k).name],'NumHeaderLines',1);  %
    Polhemus=table2array(Polhemus_temp);


    % i need to do some changes here for some particpants due to recording
    % issues
    if strcmp(f(k).name, 'polhemus_001_20250603_131823.csv') % there are some extra numbers 3 and 4 in the first col and -1 in the 3rd col  maybe i left too mcuh time to record while having closed the polhemus
        % Keep only rows where the first column is 1 or 2
        % - AND column 3 is 0 or 1
        Polhemus = Polhemus( ...
            (Polhemus(:,1) == 1 | Polhemus(:,1) == 2) & ...
            (Polhemus(:,3) == 0 | Polhemus(:,3) == 1), :);
    end
    %Exp

       clear  Polhemus_temp 



    % Separate people,
    if mod(size(Polhemus,1),2) ~= 0
        Polhemus(end,:)=0;
    end

    if Polhemus(end, 1)==0
        Polhemus(end,:) = [];
    end

    polhemus_raw_1{k} = Polhemus(1:2:end, :);
    polhemus_raw_2{k}= Polhemus(2:2:end, :);

    %recording specific correction
    if strcmp(f(k).name, 'polhemus_001_20250613_131816.csv')
        [~, ia1, ~] = unique(polhemus_raw_1{k}(:,2), 'first');  % First occurrences only
        polhemus_raw_1{k} = polhemus_raw_1{k}(ia1, :);

        [~, ia2, ~] = unique(polhemus_raw_2{k}(:,2), 'first');  % First occurrences only
        polhemus_raw_2{k} = polhemus_raw_2{k}(ia2, :);

    end
    % Save filename for reference
    Polhemus_filename{k} = f(k).name;
    clear Polhemus_temp
    clear Behavioral_temp
    clear Polhemus
end




%% 
% PREPROCESSING STARTS HERE
% LOAD BEHAVIORAL FILE AND SAVE IT IN CONDITIONS

%mine
 %datapath_behav='C:\Users\kyvko\Dropbox\DTU Copenhagen\Experiments\SecondMirror\Data\Behavioral\\';

 %yours
datapath_behav='your path with behavioral txt data';

f_behave=dir(fullfile(datapath_behav,'logfile_trialorder*.txt'));


for k=1:length(f_behave)

    Behavioral_temp= readtable([datapath_behav f_behave(k).name]);
    Condition{k}=table2array(Behavioral_temp(:,"Var1"));

    clear Behavioral_temp

end



%% 
% INTERPOLATION + TRIAL EXTRACTION + CONDITION
%mine
%RawDataPath='C:\Users\kyvko\Dropbox\DTU Copenhagen\Analysis\SecondMirror\\'

%yours
%mine
RawDataPath='your path with raw data'

load ([FilterDataPath 'Raw_Data_Polhemus_Mirror2.mat'])

numSub=length(polhemus_raw_1)

for k=1:numSub

    % Get raw data for this participant
    polhemus_1 = polhemus_raw_1{k};
    polhemus_2 = polhemus_raw_2{k};


    % interpolate interp1 (method 'pchip')
    sample_extended=[(polhemus_1(1,2):polhemus_1(end,2))];

    % interp trigger and coordinates
    for i=2:size(polhemus_1,2)
        polhemus_1_inter(:,i) = interp1(polhemus_1(:,2),polhemus_1(:,i),sample_extended,'pchip')';
        polhemus_2_inter(:,i) = interp1(polhemus_2(:,2),polhemus_2(:,i),sample_extended,'pchip')';

    end

    %add also the sample numbers after interp
    polhemus_1_inter(:,2)=sample_extended';
    polhemus_2_inter(:,2)=sample_extended';


    polhemus_1_inter(:,1)=[1:size(polhemus_1_inter,1)]';
    polhemus_2_inter(:,1)=[1:size(polhemus_2_inter,1)]';



    % first 0 corresponds to the beginning of experiment
    ind_exp_start= find(polhemus_1_inter(:,3)==0, 1, 'first')


    % last 1 corresponds to the end of experiment
    ind_exp_end = find(polhemus_1_inter(:,3)==0, 1, 'last')


    % put in a cell array the data only when the condition is active

    % these are all for finding when the condition is active
    for i=1:size(Condition{k},1)

        if i==1
            % for the first condition as initial point for
            % searching the beginning of the experiment is used
            index_to_check_start=ind_exp_start;

        else
            % for the other conditions as initial point for
            % searching the index after the end of previous trial is used
            index_to_check_start=index_end(i-1)+index_to_check_start+index_start(i-1)-2;

        end

        index_start(i) = find(polhemus_1_inter(index_to_check_start:ind_exp_end,3)==1, 1, 'first');

        index_end(i)= find(polhemus_1_inter(index_start(i)+index_to_check_start-1:ind_exp_end,3)==0, 1, 'first');

        polhemus_data_1{i} = polhemus_1_inter (index_start(i)+index_to_check_start-1:index_end(i)+index_to_check_start+index_start(i)-3,:);

        polhemus_data_2{i} = polhemus_2_inter (index_start(i)+index_to_check_start-1:index_end(i)+index_to_check_start+index_start(i)-3,:);

    end



    %  add the condition from the behavioral csv in the data
    for i=1:length(Condition{k})

        for j=1:size(polhemus_data_1{i},1)

            polhemus_data_1{i}(:,7)=Condition{k}(i);
            polhemus_data_2{i}(:,7)=Condition{k}(i);

        end

    end


    Polhemus_allCouples_Subjects_1{k} = polhemus_data_1;
    Polhemus_allCouples_Subjects_2 {k}= polhemus_data_2;


    clear polhemus_data_1 polhemus_data_2 tms polhemus_1_inter polhemus_2_inter polhemus_1 polhemus_2 Polhemus

end


%%
% calc 1D position
fprintf('\n\tCalculating 1D position data...');
for k=1:size(Polhemus_allCouples_Subjects_1,2)
    for i=1:size(Polhemus_allCouples_Subjects_1{k},2)

        Polhemus_allCouples_Subjects_1{k}{i}(:,8) = sqrt(power(Polhemus_allCouples_Subjects_1{k}{i}(:,4),2) + power(Polhemus_allCouples_Subjects_1{k}{i}(:,5),2) + power(Polhemus_allCouples_Subjects_1{k}{i}(:,6),2));
        Polhemus_allCouples_Subjects_2{k}{i}(:,8) = sqrt(power(Polhemus_allCouples_Subjects_2{k}{i}(:,4),2) + power(Polhemus_allCouples_Subjects_2{k}{i}(:,5),2) + power(Polhemus_allCouples_Subjects_2{k}{i}(:,6),2));
    end
end
fprintf('\t\tdone');


%% Filter


sampling_rate=240;
window_time=0.2; 
windowSize = sampling_rate * window_time; % it has to be an integer, otherwise try another window_time

b = (1/windowSize)*ones(1,windowSize);
a = 1;



polhemus_data_filter_1=Polhemus_allCouples_Subjects_1;
polhemus_data_filter_2=Polhemus_allCouples_Subjects_2;
% we do it for the x,y,z coordinates and distance data
for k=1:size(polhemus_data_filter_1,2)
    for i=1:size(polhemus_data_filter_1{k},2)
        polhemus_data_filter_1{k}{i}(:,[4:6 8]) = filter(b,a,Polhemus_allCouples_Subjects_1{k}{i}(:,[4:6 8]));
        polhemus_data_filter_2{k}{i}(:,[4:6 8]) = filter(b,a,Polhemus_allCouples_Subjects_2{k}{i}(:,[4:6 8]));


    end

end

%% SAVE THE DATA AS I SAVED IT IN THE PAST IN CSV FILES FOR YOU. EACH FILE FOR SEPARATE SUBJECT AND TRIAL, IN CASE YOU WANT IT LIKE THIS (it is not needed for running the PLV)

numSubjects = length(polhemus_data_filter_2);

% Main output directory
outputDir = 'your path';

% outputDir = 'C:\Users\kyvko\Dropbox\DTU Copenhagen\Experiments\SecondMirror\Data\polhemusForShane\Second_Subject\';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

for subj = 1:numSubjects
    % Create subject folder
    subjFolder = fullfile(outputDir, sprintf('Subject_%02d', subj+200));
    if ~exist(subjFolder, 'dir')
        mkdir(subjFolder);
    end
    
    % Get trials for this subject
    trials = polhemus_data_filter_2{subj};
    
    for t = 1:length(trials)
        trialData = trials{t};
        
        % Define filename
        fileName = sprintf('Trial_%02d.csv', t);
        filePath = fullfile(subjFolder, fileName);
        
        % Save to CSV
        writematrix(trialData, filePath);
    end
end

