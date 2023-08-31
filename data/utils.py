import codecs
import copy
import os

from typing import List, Optional, Union

from nltk import sent_tokenize


def print_keys_recursively(obj: Union[list, dict, str], tab: int = 0):
    """Print data structure recursively

    Args:
        obj (Union[list, dict, str]): data
        tab (int, optional): tab number to indicate structure
    """
    # TODO: will be executed in convert.py for showing structure of dataset
    if isinstance(obj, list):
        print_keys_recursively(obj[0], tab)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            print('\t' * tab, k)
            if v:
                print_keys_recursively(v, tab+1)


def make_ems(
    evidence: dict,
    qmsum: dict,
    dialogue_act_path: str,
    data_name: str
) -> dict:
    """Construct ExplainMeetSum with SummaryEvidence and QMSum

    Construct ExplainMeetSum by merging QMSum dataset and SummaryEvidence.
    `evidence` will be converted into ExplainMeetSum. After adding
    `meeting_transcripts` and `answer`, `topic_list` in `qmsum` to `evidence`,
    the converting process will be done.

    Args:
        evidence (dict): Each data of SummaryEvidence
        qmsum (dict): Each data of QMSum
        dialogue_act_path (str): Root directory of dialogue_act files
        data_name (str): Each data name

    Note:
        Answers of queries in QMSum are already splited in SummaryEvidence.
        You can see how we split sentences on `split_query_answer` in "sentence_split.py"

        Although using `split_query_answer`, there's wrongly splited answers still remain.
        That answers and answers that have no **evidence** will be merged previous sentence.
        Merging will be executed by using `answer_sentence_index` in SummaryEvidence
        while making ExplainMeetSum.

    Returns:
        dict: one ExplainMeetSum data
    """

    # ExplainMeetSum
    ems = dict()

    da_dir = os.path.join(
        dialogue_act_path,
        'data/meeting',
        domain[domain.find("(") + 1: domain.find(")")],
        f'{data_name}.da'
    ) if (domain := evidence['domain']) != 'Committee' else None

    ems['domain'] = evidence['domain']
    ems['topic_list'] = qmsum['topic_list']

    # TODO: will be implemented soon...
    ems['explainable_ami'] = evidence['ami']
    ems['explainable_icsi'] = evidence['icsi']

    query_list = _merge_query_list(evidence, qmsum)
    ems['explainable_qmsum'] = dict()
    ems['explainable_qmsum']['general_query_list'] = query_list['general_query_list']
    ems['explainable_qmsum']['specific_query_list'] = query_list['specific_query_list']

    ems['meeting_transcripts'] = _split_sent_in_transcripts(qmsum, da_dir)

    return ems


def _merge_two_dicts(first: dict, second: dict) -> dict:
    """Merge two dictionaries
    """
    return {**first, **second}


def _merge_query_list(evidence: dict, qmsum: dict) -> dict:
    """Merge query lists in `evidence` and `qmsum`

    Args:
        evidence (dict): Each data of SummaryEvidence
        qmsum (dict): Each data of QMSum
    """

    # general_query_list or specific_query_list
    converted_query_list = dict()

    for k, v in evidence['qmsum'].items():
        query_list = copy.deepcopy(v)
        for i, query in enumerate(query_list):

            query['answer'] = ' '.join(query['tokenized_answer'])

            if k == 'specific_query_list':
                query['relevant_text_span'] = qmsum['specific_query_list'][i]['relevant_text_span']

            query['explainable_answer'] = [
                {
                    'answer_sentence': answer_sentence,
                    'evidence': query['evidence'][i],
                    'answer_sentence_index': query['answer_sentence_index'][i]
                }
                for i, answer_sentence in enumerate(query['tokenized_answer'])
            ]
            query.pop('evidence')
            query.pop('tokenized_answer')
            query.pop('answer_sentence_index')

        converted_query_list[k] = query_list

    return converted_query_list


def _split_sent_in_transcripts(
    qmsum: dict,
    dialogue_act_file: Optional[str]
) -> list:
    """Split sentences of `meeting_transcripts` in QMSum
    and save it into `evidence`

    Args:
        qmsum (dict): original QMSum data
        dialogue_act_file (Optional[str], optional): path of *.da file.

    Note:
        Two way of split exist.

        1) 'committee' type dataset:
            split sentences with `nltk.sent_tokenize`

        2) 'ami', 'icsi' type dataset:
            split sentences with original files that specify each utterance
            - You should download *.da files first to split sentences correctly.

    Returns:
        list: splited `meeting_transcripts`
    """

    qmsum_transcripts = qmsum['meeting_transcripts']
    merged_transcripts = list()

    # True: Committe / False: Ami, Icsi
    flag = not bool(dialogue_act_file)

    dialogue_transcripts = [None for _ in range(len(qmsum_transcripts))] if flag \
        else _split_sent_with_file(dialogue_act_file)

    for i, (_dialogues, qmsum) in enumerate(zip(dialogue_transcripts, qmsum_transcripts)):
        dialogues = sent_tokenize(qmsum['content']) if flag else _dialogues
        tmp_list = [
            {
                'turn_index': i,
                'sent_index': j,
                'dialogue_sentence': dialogue if flag else dialogue['dialogue_sentence'],
                'dialogue-act_id': None if flag else dialogue['dialogue-act_id']
            }
            for j, dialogue in enumerate(dialogues)
        ]
        merged_transcripts.append(
            _merge_two_dicts(qmsum, {'sentence_level_content': tmp_list})
        )

    return merged_transcripts


def _split_sent_with_file(dialogue_act_file: str) -> List[dict]:
    """Split sentences with original file of dataset

    Args:
        dialogue_act_file (str): root directory of *.da files.

    Raises:
        FileNotFoundError:

    Returns:
        List[dict]: original meeting transcript seperated
            with speaker and each utterance.
    """

    if dialogue_act_file and not os.path.isfile(dialogue_act_file):
        raise FileNotFoundError(
            f"File doesn't exist. Check if you place `{dialogue_act_file}` on right place"
        )

    dialogue_transcript = list()
    prev_speaker = ""

    with codecs.open(dialogue_act_file) as f:
        for line in f:
            if line := line.strip():
                parts = line.split("\t")

                dialog_act = parts[0]
                speaker = parts[3]

                # Case without utterance
                try:
                    utterance = parts[8]
                except IndexError:
                    utterance = ""

            # other speaker
            if prev_speaker != speaker:
                dialogue_transcript.append([{'dialogue_sentence': utterance, 'dialogue-act_id': dialog_act}])
            # same speaker
            else:
                dialogue_transcript[-1].append({'dialogue_sentence': utterance, 'dialogue-act_id': dialog_act})
            prev_speaker = speaker

    return dialogue_transcript
