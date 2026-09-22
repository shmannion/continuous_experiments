%% before i did not use the filter data
clear all
close all
%FilterDataPath='your path with filtered data'

FilterDataPath='C:\Users\kyvko\Dropbox\DTU Copenhagen\Analysis\SecondMirror\\'
load ([FilterDataPath 'Filter_Data_Polhemus_Mirror2.mat'])
k=1
Fs=240;
col=8;
clear polhemus_hilbert_1 polhemus_hilbert_2 sigphase1 sigphase2 dtheta x


sliding_window = 2.5;        % step between window starts (in seconds)
win_analysis_sec = 5;     % size of each analysis window (in seconds)
sec_per_trial = 30;        % total trial duration (in seconds)

% Calculate the latest valid starting point to stay within trial duration
last_valid_start = sec_per_trial - win_analysis_sec;

% Create the analysis start times
analysis_start = 0:sliding_window:last_valid_start;



no_trials=length(polhemus_data_filter_1{k});


no_Subj = length(polhemus_data_filter_1);


PLV_win=NaN(no_Subj, no_trials,length(analysis_start));

occur_synchro =NaN(no_Subj, no_trials);

%%
clear polhemus_hilbert_1 polhemus_hilbert_2 sigphase1 sigphase2 dtheta x
condition_names = {'Solo', 'Spontaneous', 'Self Both', 'Other Both', 'Self 1 Other 2', 'Self 2 Other 1'};

for k=1:no_Subj
    for i=1:length(polhemus_data_filter_1{k})
        occur_synchro(k,i)=0;
        if  length(polhemus_data_filter_1{k}{i})==0
            continue
        else

            tms = (0:numel(polhemus_data_filter_1{k}{i}(:,col))-1)/Fs;
        end
        for j=1:length(analysis_start)
            segment= dsearchn(tms',analysis_start(j)+0.25);

            segment_final= dsearchn(tms',analysis_start(j)+win_analysis_sec');
            tms_segment = tms(segment:segment_final);

            if size(tms_segment)==1
                break
            end

            % i am not sure why i was doing this part with the if and j
            % maybe it is not needed
            if j>1
                if length(tms_segment)~=length_to_compare
                    break
                end
            end
            length_to_compare=length(polhemus_data_filter_1{k}{i}(segment:segment_final, col));


            polhemus_hilbert_1{k}{i}(:,j) = hilbert(zscore(polhemus_data_filter_1{k}{i}(segment:segment_final, col)));
            polhemus_hilbert_2{k}{i}(:,j) = hilbert(zscore(polhemus_data_filter_2{k}{i}(segment:segment_final, col)));
            sigphase1{k}{i}(:,j) = atan2(imag(polhemus_hilbert_1{k}{i}(:,j)),real(polhemus_hilbert_1{k}{i}(:,j)));
            sigphase2{k}{i}(:,j) = atan2(imag(polhemus_hilbert_2{k}{i}(:,j)),real(polhemus_hilbert_2{k}{i}(:,j)));
          


            dtheta{k}{i}(:,j) = sigphase2{k}{i}(:,j)-sigphase1{k}{i}(:,j);
            x{k}{i}(:,j)=exp(1i*dtheta{k}{i}(:,j));
            PLV_win(k,i,j) = abs(nanmean(x{k}{i}(:,j)));


            if   PLV_win(k,i,j) >0.5 % it means that the segment is synchronized
                occur_synchro(k,i) = occur_synchro(k,i)+1;
            end

        end

    end
end


%%
% if you want to have an average value over all the windows
PLV_averageSeg = nanmean(PLV_win,3);
for k=1:size(PLV_averageSeg,1)

    ind_solo=find(Condition{k}==1);
    PLV_Solo(k) = nanmean(PLV_averageSeg(k, ind_solo), 2);
    occur_Solo(k)=nanmean(occur_synchro(k,ind_solo ), 2);

    ind_Spont=find(Condition{k}==2);
    PLV_Spont(k) = nanmean(PLV_averageSeg(k, ind_Spont), 2);
    occur_Spont(k)=nanmean(occur_synchro(k, ind_Spont), 2);

    ind_Self=find(Condition{k}==3);
    PLV_Self(k) = nanmean(PLV_averageSeg(k, ind_Self), 2);
    occur_Self(k)=nanmean(occur_synchro(k, ind_Self), 2);

    ind_Other=find(Condition{k}==4);
    PLV_Other(k) = nanmean(PLV_averageSeg(k, ind_Other), 2);
    occur_Other(k)=nanmean(occur_synchro(k, ind_Other), 2);

    ind_Self_1_Other_2=find(Condition{k}==5);
    PLV_Self_1_Other_2(k) = nanmean(PLV_averageSeg(k, ind_Self_1_Other_2), 2);
    occur_Self_1_Other_2(k)=nanmean(occur_synchro(k, ind_Self_1_Other_2), 2);

    ind_Self_2_Other_1=find(Condition{k}==6);
    PLV_Self_2_Other_1(k) = nanmean(PLV_averageSeg(k, ind_Self_2_Other_1), 2);
    occur_Self_2_Other_1(k)=nanmean(occur_synchro(k, ind_Self_2_Other_1), 2);

    ind_Self_Other=[ind_Self_1_Other_2 ind_Self_2_Other_1];
    PLV_Self_Other(k) = nanmean(PLV_averageSeg(k, ind_Self_Other), 2);
    occur_Self_Other(k)=nanmean(occur_synchro(k, ind_Self_Other), 2);

end



%%

final_PLV = [PLV_Solo'; PLV_Spont' ;PLV_Self';PLV_Other';PLV_Self_Other'];

for i=1:length(PLV_Solo)
F{i,1} = ['Solo'] ;
end
for i=(length(PLV_Solo)+1):(2*length(PLV_Solo))
  F{i,1} = ['Spontaneous'] ;
end  
for i=(2*length(PLV_Solo)+1):(3*length(PLV_Solo))
  F{i,1} = ['Self'] ;
end  
for i=(3*length(PLV_Solo)+1):(4*length(PLV_Solo))
  F{i,1} = ['Other'] ;
end  
for i=(4*length(PLV_Solo)+1):(5*length(PLV_Solo))
  F{i,1} = ['Self_Other'] ;
end 

for i=1:length(F)
F{i,2} = ['PLV'] ;
end
for i=1:length(F)
F{i,3} = final_PLV(i);
end


IDs = (1:length(PLV_Solo))'; % participant IDs

ID = repmat(IDs,5,1);
for i=1:length(F)
F{i,4} = ID(i);
end
F_final={'Condition', 'PLVIndex', 'Values', 'ID'}
F_final=[F_final;F];

% Convert cell to a table and use first row as variable names
T = cell2table(F_final(2:end,:),'VariableNames',F_final(1,:))

% Write the table to a CSV file
writetable(T,'PLV_42_windows_5s_try.csv')




