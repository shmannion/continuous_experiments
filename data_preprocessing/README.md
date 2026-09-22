# Scripts for getting from raw data files to files ready-to-use for analysis and modelling.



- marius_data_preprocessing.py: script to take the raw data files from marius' experiment, separate them
  by participant, condition, and trial to make them easier to analyse. More detail in .py file.

- marius_data_coordinates.py: script to take the x,y,z co-ordinates and project them onto the vector corresponding to 
  the rail. Uses functions from marius_data_pairing.py in common/ which are for pairing up the data from different 
  trials.

## N.B. Marius naming conventions.
text_condition{1} = 'BASELINE';
text_condition{2} = 'A+B MOVE ALONE';
text_condition{3} = 'A+B MOVE TOGETHER';
text_condition{4} = 'A MOVE - B OBSERVE';
text_condition{6} = 'B MOVE - A OBSERVE';
text_condition{5} = 'A MOVE - B IMITATE';
text_condition{7} = 'B MOVE - A IMITATE';
text_condition{8} = 'Imitate Dot Movement';

As well as this
% ANALTSIS SETTINGS
% 1 => rest
% 2 => uncoupled
% 3 => coupled
% 4 => observe, actor       ppn1 6, ppn2 4
% 5 => observe, observer    ppn1 4, ppn2 6
% 6 => imitate, leader      ppn1 7, ppn2 5
% 7 => imitate, follower    ppn1 5, ppn2 7
% 8 => control task

I don't understand the second segment but it is included here in case I have interpretted the first incorrectly.
As I understand it, it goes as follows: 

Two experimental conditions, observer/actor and leader/follower. To get all actors, you need participant 1 from 
condition 6, and participant 2 from experiment 4. To get all observers, you need participant 2 from condition 6, 
and participant 1 from experiment 4. To get all leaders, it is participant 1 from condition 7, participant 2 from 
condition 5. To get all followers, it is participant 1 from condition 5, participant 2 from condition 7.
The A,B in Marius' naming conventions do not apply to my cleaned data.
In my datafiles, participant 1 is always participant A and 2 is always B.
As such, to get all followers, it is participant A files from 5, B from 7
to get all leaders it is A from 7, B from 5
actors is A from 6, B from 4.
observers is B from 6, A from 4

The participants do 16 trials of each condition (move alone, move together, move/observe, and lead/follow).
For 8 trials, participant A is leading, participant B is following, and the reverse for the other 8 trials.
