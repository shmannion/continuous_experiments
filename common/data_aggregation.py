from common.paths import DATA, MARIUS_RAW, MARIUS_CLEAN
from scipy.io import loadmat

def load_preprocessed_movement_data(pair, trial, participant):
    """
    get trial data for a given participant of a given pair, from the preprocessed data.
    """
    mat = loadmat(f'{DATA}/marius/processed/pair{pair:03d}_movementdata.mat',
                squeeze_me=True, struct_as_record=False)

    tr = mat['polhemus'][trial] 
    if participant == 1:
        ppn = tr.pos.ppn1
    elif participant == 2:
        ppn = tr.pos.ppn2
    else:
        ppn = [tr.pos.ppn1, tr.pos.ppn2]
    return ppn


def get_condition_files(condition: int):
    """
    """

    data_path = list(MARIUS_CLEAN.glob(f'condition_{condition}/*/*.csv'))
    print(data_path)


def get_condition_map(pair):
    """
    from the raw data trial order files, create a dict of preprocessed trial number: condition

    """
    path = f'{MARIUS_CLEAN}/trial_dicts/pair{pair:02d}_trial_order.txt'

    trial_order = {i : [] for i in range(1,9)}
    with open(path) as file:
       for idx, line in enumerate(file):
           l = line.strip().split(',')
           condition_no = int(l[0]) 
           trial_order[condition_no].append(idx + 1)
    return trial_order



if __name__ == "__main__":
    
    pair = 3
    trial_order = get_condition_map(pair)
    print(trial_order)
