import argparse
import glob
import json
import os

from tqdm import tqdm

from utils import make_ems


def convert_ems(data_type: str, path: argparse.Namespace) -> None:
    """Covert SummaryEvidence into ExplainMeetSum

    Args:
        data_type (str): data type between ['train', 'val', 'test']
        path (argparse.Namespace): group of path
            1. SummaryEvidence/ directory
            2. QMSum/ directory
            3. acl2018_abssumm/ directory for dialogue_act files
            4. ExplainMeetSum/ directory

    Raises:
        FileNotFoundError: Raise error if there's no matching QMSum file
    """

    # directory where to save `ExplainMeetSum` dataset
    save_dir = os.path.join(path.save_dir, data_type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    evidence_list = glob.glob(os.path.join(path.evidence, data_type, '*.json'))

    # check whether path is correct
    # assert len(evidence_list) > 0, f"There's no SummaryEvidence file on {os.path.join(path.evidence, data_type)}"

    for evidence_file in tqdm(evidence_list, desc=f'{data_type: <6}'):
        data_name = os.path.basename(evidence_file).split('.')[0]

        # QMSum .json file matching with SummaryEvidence .json file
        qms_file = os.path.join(path.qmsum, 'data/ALL', data_type, f'{data_name}.json')
        if not os.path.isfile(qms_file):
            raise FileNotFoundError(f"`{qms_file}` doesn't exist.")

        with open(evidence_file, 'rt', encoding='UTF8') as f:
            evidence = json.load(f)

        with open(qms_file, 'rt', encoding='UTF8') as f:
            qmsum = json.load(f)

        ems = make_ems(evidence, qmsum, path.dialogue_act, data_name)

        ems_dir = os.path.join(save_dir, f'{data_name}.json')
        with open(ems_dir, 'w+') as f:
            # Save each ExplaineMeetSum data
            json.dump(ems, f, indent=4)


def get_path():
    """Return group of path
    """
    parser = argparse.ArgumentParser(prog='group of paths')
    parser.add_argument('--evidence',
                        type=str,
                        default='data/SummaryEvidence',
                        help='path of SummaryEvidence dataset')

    # https://github.com/Yale-LILY/QMSum
    parser.add_argument('--qmsum',
                        type=str,
                        default='data/QMSum',
                        help='path of QMSum\'s root directory')

    # https://bitbucket.org/dascim/acl2018_abssumm
    parser.add_argument('--dialogue_act',
                        type=str,
                        default='data/acl2018_abssumm',
                        help='path of acl2018_abssumm\'s root directory')

    parser.add_argument('--save_dir',
                        type=str,
                        default='data/ExplainMeetSum',
                        help='path where to save ExplainMeetSum dataset')
    args = parser.parse_args()
    return args


def check_sanity(ems_path: str):
    """Check sanity
    """
    data_length = {
        'train': 161,
        'val': 35,
        'test': 35
    }
    print()
    print('=' * 25, '\n')
    for data_type in ['train', 'val', 'test']:
        data_files = glob.glob(f'{ems_path}/{data_type}/*.json')

        print(f'{data_type :>8}: {len(data_files) :>3} / {data_length[data_type] :>3}')

    print()
    print('=' * 25, '\n')


if __name__ == '__main__':
    path = get_path()

    print()

    for data_type in ['train', 'val', 'test']:
        convert_ems(data_type, path)

    check_sanity(path.save_dir)

    print(f"🎉 `ExplainMeetSum` dataset is saved on {os.path.abspath(path.save_dir)}")
