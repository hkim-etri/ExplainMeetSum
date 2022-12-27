"""Module providingFunction printing python version."""
import codecs
from xml.dom import minidom
import os

import json
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge import Rouge
from openpyxl import load_workbook
import xlsxwriter
import copy
from utils.utils import rouge_with_pyrouge
import numpy as np

from sentence_transformers import SentenceTransformer, util
sentence_transformers_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_id(text):
    """get general id"""
    cleaned_text = text.split('-')
    id = {'first-index': int(cleaned_text[0]), 'second-index': int(cleaned_text[1]), 'text':text}
    return id

def specific_get_id(text):
    """get specific id"""
    assert text[:2] == '<S'
    assert text[-1] == '>'
    cleaned_text = text[2:-1]
    id = get_id(cleaned_text)
    new_id = {'query-index': id['first-index'], 'sentence-index': id['second-index'], 'text':text}
    return new_id

def generalA_get_id(text):
    """get generalA id"""
    assert text[:3] == '<GA'
    assert text[-1] == '>'
    cleaned_text = text[3:-1]

    assert cleaned_text[0] == '-'
    cleaned_text = cleaned_text[1:]
    new_id = {'query-index': 'A', 'sentence-index': int(cleaned_text), 'text':text}
    return new_id

def generalB_get_id(text, length):
    """get generalB id"""
    assert text[:3] == '<GB'
    assert text[-1] == '>'
    if length == 1:
        cleaned_text = text[3:-1]
    else:
        assert length < 10
        cleaned_text = text[4:-1]
    assert cleaned_text[0] == '-'
    cleaned_text = cleaned_text[1:]
    new_id = {'query-index': 'B', 'sentence-index': int(cleaned_text), 'text':text}
    return new_id


def rouge(dec, ref):
    """get rouge scoure"""
    if dec == '' or ref == '':
        return 0.0
    rouge = Rouge()
    scores = rouge.get_scores(dec, ref)
    return (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3

def get_file_names(path):
    """get file names"""
    names = []
    with codecs.open(path, "r", "utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                names.append(line)
    return names

def get_project_dir():
    """get project dir"""

    current_dir = os.path.abspath(os.path.dirname(__file__))
    return current_dir

def load_GeneralA(excel_data, file_name, qmsum_data, data_type, all_answers, all_ambiguous, extracted_merges):
    """load GeneralA query"""

    name = "generalA"
    query_name = "{}.generalA".format(file_name.split('.')[0])

    print(query_name)

    general_excel_data = excel_data[name]

    answer_starts = []
    answer_info = {}
    answer_sent_index = 0
    for i, rows in enumerate(general_excel_data.rows):
        value = rows[2].value
        if type(value) != str:
            continue
        elif value[:3] != '<GA':
            continue
        else:
            answer_id = generalA_get_id(value)
            assert answer_id['sentence-index'] == answer_sent_index
            answer_starts.append(i)
            answer_info[i] = answer_id
            answer_sent_index += 1

    check_answer_start = False
    answers = {'answer_info': {'sent':qmsum_data['generalA_summary'], 'file_name':file_name}, 'sent':{}}
    prev_answer_index = answer_starts[0]
    prev_answer_text = ""
    for i, rows in enumerate(general_excel_data.rows):
        if i in answer_starts:
            check_answer_start = True
            prev_answer_index = i
        else:
            if check_answer_start == False:
                continue

        if rows[4].value == None:
            check_answer_start = False
            continue

        answer_id = answer_info[prev_answer_index]

        if answer_id['sentence-index'] not in answers['sent'].keys():
            answers['sent'][answer_id['sentence-index']] = []
        if prev_answer_index == i:
            assert rows[3].value != None
            prev_answer_text = rows[3].value

        if rows[4].value.split()[0] == 'Ambiguous':
            if rows[4].value.split()[1] != '(2.all)':
                assert rows[7].value != None



            if query_name not in all_ambiguous:
                all_ambiguous[query_name] = []
            entry = {'answer_id':answer_id, 'tag':rows[4].value.split()[1], 'text':rows[7].value}
            all_ambiguous[query_name].append(entry)
            continue
        elif rows[4].value.split()[0] == '<Merge>':
            print("\t({}) <Merge> :  {}  ".format(answer_id['text'], rows[5].value))

            if query_name not in extracted_merges:
                extracted_merges[query_name] = []
            entry = {'answer_id':answer_id, 'tag':rows[4].value, 'value':rows[5].value, 'data_type': data_type}
            extracted_merges[query_name].append(entry)

            continue
        elif rows[4].value.split()[0] == '<Delete>':
            assert rows[5].value == None
            print("\t({}) <Delete>".format(answer_id['text']))

            if query_name not in extracted_merges:
                extracted_merges[query_name] = []
            entry = {'answer_id':answer_id, 'tag':rows[4].value, 'value':rows[5].value, 'data_type': data_type}
            extracted_merges[query_name].append(entry)
            continue
        else:
            linking_index = get_id(rows[4].value)
        key_info = rows[5].value

        if data_type == 'product' or data_type == 'academic':
            utter_i = 'da_i'
            content_utter = 'content_da'
        else:
            utter_i = 'sent_i'
            content_utter = 'content_check_sent'

        evidence_sent_text = rows[6].value
        linking_speaker = qmsum_data['meeting_transcripts'][linking_index['first-index']]['speaker']
        f_i = linking_index['first-index']
        s_i = linking_index['second-index']
        linking_text = qmsum_data['meeting_transcripts'][f_i][content_utter][s_i]['text']
        # assert "<{}> : {}".format(linking_speaker, linking_text) == evidence_sent_text
        if "<{}> : {}".format(linking_speaker, linking_text) != evidence_sent_text:
            print("<{}> : {}".format(linking_speaker, linking_text))
            print(evidence_sent_text)
            if linking_text == evidence_sent_text:
                evidence_sent_text = "<{}> : {}".format(linking_speaker, linking_text)
            else:
                print("Not equal evidence sentence : {}".format(rows[4].value))
                assert False
        # assert "<{}> : {}".format(linking_speaker, linking_text) == evidence_sent_text

        evidence_sent_text = "< {} > : {}".format(linking_speaker, linking_text)

        if rows[7].value == None:
            related_part = []
        else:
            tmp_list = rows[7].value.split('\n')
            related_part = [r_t.strip() for r_t in tmp_list]

        additional_tags = []
        if key_info.split()[0] == 'X':
            # assert rows[8].value != None
            if rows[8].value != None:
                additional_tags.append(rows[8].value)
            if rows[9].value != None:
                additional_tags.append(rows[9].value)
            if rows[10].value != None:
                additional_tags.append(rows[10].value)
        if rows[11].value != None:
            comment = rows[11].value
        else:
            comment = ""

        tag = {}
        tag['query_id'] = query_name
        tag['answer_sent_index'] = answer_id['text']
        tag['linking'] = rows[4].value
        tag['answer_sent_text'] = prev_answer_text
        tag['key_tag'] = key_info
        tag['evidence_sent_text'] = evidence_sent_text
        tag['linking_speaker'] = linking_speaker
        tag['linking_text'] = linking_text
        tag['related_part'] = related_part
        tag['additional_tags'] = additional_tags
        tag['comment'] = comment
        tag['file_name'] = file_name
        tag['data_type'] = data_type        
        
        answers['sent'][answer_id['sentence-index']].append(tag)

    answers['data_type'] = data_type
    all_answers[query_name] = answers

    return



def load_GeneralB(excel_data, file_name, qmsum_data, data_type, all_answers, all_ambiguous, extracted_merges):
    """load GeneralB query"""
    for g_i, general_query in enumerate(qmsum_data['general_query_list']):
        if len(qmsum_data['general_query_list']) == 1:
            name = "generalB"
            query_name = "{}.generalB".format(file_name.split('.')[0])

        else:
            name = "generalB{}".format(g_i)
            query_name = "{}.generalB{}".format(file_name.split('.')[0], g_i)

        print(query_name)
        general_excel_data = excel_data[name]

        answer_starts = []
        answer_info = {}
        answer_sent_index = 0
        for i, rows in enumerate(general_excel_data.rows):
            value = rows[2].value
            if type(value) != str:
                continue
            elif value[:3] != '<GB':
                continue
            else:
                answer_id = generalB_get_id(value, len(qmsum_data['general_query_list']))
                assert answer_id['sentence-index'] == answer_sent_index
                answer_starts.append(i)
                answer_info[i] = answer_id
                answer_sent_index += 1

        check_answer_start = False
        answers = {'answer_info': {'sent':general_query['answer_sent_based'], 'file_name':file_name}, 'sent':{}}
        prev_answer_index = answer_starts[0]
        prev_answer_text = ""
        for i, rows in enumerate(general_excel_data.rows):
            if i in answer_starts:
                check_answer_start = True
                prev_answer_index = i
            else:
                if check_answer_start == False:
                    continue

            if rows[4].value == None:
                check_answer_start = False
                continue

            answer_id = answer_info[prev_answer_index]

            if answer_id['sentence-index'] not in answers['sent'].keys():
                answers['sent'][answer_id['sentence-index']] = []
            if prev_answer_index == i:
                assert rows[3].value != None
                prev_answer_text = rows[3].value

            if rows[4].value.split()[0] == 'Ambiguous':
                if rows[4].value.split()[1] != '(2.all)':
                    assert rows[7].value != None

                if query_name not in all_ambiguous:
                    all_ambiguous[query_name] = []

                entry = {'answer_id':answer_id, 'tag':rows[4].value.split()[1], 'text':rows[7].value}
                all_ambiguous[query_name].append(entry)
                continue
            elif rows[4].value.split()[0] == '<Merge>':
                print("\t({}) <Merge> :  {}  ".format(answer_id['text'], rows[5].value))

                if query_name not in extracted_merges:
                    extracted_merges[query_name] = []
                entry = {'answer_id':answer_id, 'tag':rows[4].value, 'value':rows[5].value, 'data_type': data_type}
                extracted_merges[query_name].append(entry)

                continue
            elif rows[4].value.split()[0] == '<Delete>':
                assert rows[5].value == None
                print("\t({}) <Delete>".format(answer_id['text']))

                if query_name not in extracted_merges:
                    extracted_merges[query_name] = []
                entry = {'answer_id':answer_id, 'tag':rows[4].value, 'value':rows[5].value, 'data_type': data_type}
                extracted_merges[query_name].append(entry)
                continue
            else:
                linking_index = get_id(rows[4].value)
            key_info = rows[5].value

            if data_type == 'product' or data_type == 'academic':
                utter_i = 'da_i'
                content_utter = 'content_da'
            else:
                utter_i = 'sent_i'
                content_utter = 'content_check_sent'

            evidence_sent_text = rows[6].value
            linking_speaker = qmsum_data['meeting_transcripts'][linking_index['first-index']]['speaker']
            f_i = linking_index['first-index']
            s_i = linking_index['second-index']
            linking_text = qmsum_data['meeting_transcripts'][f_i][content_utter][s_i]['text']            
            # assert "<{}> : {}".format(linking_speaker, linking_text) == evidence_sent_text
            if "<{}> : {}".format(linking_speaker, linking_text) != evidence_sent_text:
                print("<{}> : {}".format(linking_speaker, linking_text))
                print(evidence_sent_text)
                if linking_text == evidence_sent_text:
                    evidence_sent_text = "<{}> : {}".format(linking_speaker, linking_text)
                else:
                    print("Not equal evidence sentence : {}".format(rows[4].value))
                    assert False

            evidence_sent_text = "< {} > : {}".format(linking_speaker, linking_text)

            if rows[7].value == None:
                related_part = []
            else:
                tmp_list = rows[7].value.split('\n')
                related_part = [r_t.strip() for r_t in tmp_list]

            additional_tags = []
            if key_info.split()[0] == 'X':
                # assert rows[8].value != None
                if rows[8].value != None:
                    additional_tags.append(rows[8].value)
                if rows[9].value != None:
                    additional_tags.append(rows[9].value)
                if rows[10].value != None:
                    additional_tags.append(rows[10].value)
            if rows[11].value != None:
                comment = rows[11].value
            else:
                comment = ""

            tag = {}
            tag['query_id'] = query_name
            tag['answer_sent_index'] = answer_id['text']
            tag['linking'] = rows[4].value
            tag['answer_sent_text'] = prev_answer_text
            tag['key_tag'] = key_info
            tag['evidence_sent_text'] = evidence_sent_text
            tag['linking_speaker'] = linking_speaker
            tag['linking_text'] = linking_text
            tag['related_part'] = related_part
            tag['additional_tags'] = additional_tags
            tag['comment'] = comment
            tag['file_name'] = file_name
            tag['data_type'] = data_type        
                    
            answers['sent'][answer_id['sentence-index']].append(tag)

        answers['data_type'] = data_type
        all_answers[query_name] = answers

    return


def load_Specific(excel_data, file_name, qmsum_data, data_type, all_answers, all_ambiguous, extracted_merges):
    """load Specific query"""
    for s_i, specific_query in enumerate(qmsum_data['specific_query_list']):
        specific_excel_data = excel_data['specific{}'.format(s_i)]

        query_name = "{}.specific{}".format(file_name.split('.')[0], s_i)
        print(query_name)

        answer_starts = []
        answer_info = {}
        answer_sent_index = 0
        for i, rows in enumerate(specific_excel_data.rows):
            value = rows[2].value
            if type(value) != str:
                continue
            elif value[:2] != '<S':
                continue
            else:
                answer_id = specific_get_id(value)
                assert answer_id['query-index'] == s_i
                assert answer_id['sentence-index'] == answer_sent_index
                answer_starts.append(i)
                answer_info[i] = answer_id
                answer_sent_index += 1

        check_answer_start = False
        answers = {'answer_info': {'sent':specific_query['answer_sent_based'], 'file_name':file_name}, 'sent':{}}
        prev_answer_index = answer_starts[0]
        prev_answer_text = ""
        for i, rows in enumerate(specific_excel_data.rows):
            if i in answer_starts:
                check_answer_start = True
                prev_answer_index = i
            else:
                if check_answer_start == False:
                    continue

            if rows[4].value == None:
                check_answer_start = False
                continue

            answer_id = answer_info[prev_answer_index]

            if answer_id['sentence-index'] not in answers['sent'].keys():
                answers['sent'][answer_id['sentence-index']] = []
            if prev_answer_index == i:
                assert rows[3].value != None
                prev_answer_text = rows[3].value

            if rows[4].value.split()[0] == 'Ambiguous':
                assert rows[7].value != None

                if query_name not in all_ambiguous:
                    all_ambiguous[query_name] = []

                entry = {'answer_id':answer_id, 'tag':rows[4].value.split()[1], 'text':rows[7].value}
                all_ambiguous[query_name].append(entry)
                continue
            elif rows[4].value.split()[0] == '<Merge>':
                print("\t({}) <Merge> :  {}  ".format(answer_id['text'], rows[5].value))

                if query_name not in extracted_merges:
                    extracted_merges[query_name] = []
                entry = {'answer_id':answer_id, 'tag':rows[4].value, 'value':rows[5].value, 'data_type': data_type}
                extracted_merges[query_name].append(entry)
                continue
            elif rows[4].value.split()[0] == '<Delete>':
                assert rows[5].value == None
                print("\t({}) <Delete>".format(answer_id['text']))

                if query_name not in extracted_merges:
                    extracted_merges[query_name] = []
                entry = {'answer_id':answer_id, 'tag':rows[4].value, 'value':rows[5].value, 'data_type': data_type}
                extracted_merges[query_name].append(entry)
                continue
            else:
                linking_index = get_id(rows[4].value)
            key_info = rows[5].value

            if data_type == 'product' or data_type == 'academic':
                utter_i = 'da_i'
                content_utter = 'content_da'
            else:
                utter_i = 'sent_i'
                content_utter = 'content_check_sent'

            evidence_sent_text = rows[6].value
            linking_speaker = qmsum_data['meeting_transcripts'][linking_index['first-index']]['speaker']
            f_i = linking_index['first-index']
            s_i = linking_index['second-index']
            linking_text = qmsum_data['meeting_transcripts'][f_i][content_utter][s_i]['text']
            # assert "<{}> : {}".format(linking_speaker, linking_text) == evidence_sent_text
            if "<{}> : {}".format(linking_speaker, linking_text) != evidence_sent_text:
                print("<{}> : {}".format(linking_speaker, linking_text))
                print(evidence_sent_text)
                if linking_text == evidence_sent_text:
                    evidence_sent_text = "<{}> : {}".format(linking_speaker, linking_text)
                else:
                    print("Not equal evidence sentence : {}".format(rows[4].value))
            # assert "<{}> : {}".format(linking_speaker, linking_text) == evidence_sent_text

            evidence_sent_text = "< {} > : {}".format(linking_speaker, linking_text)

            if rows[7].value == None:
                related_part = []
            else:
                tmp_list = rows[7].value.split('\n')
                related_part = [r_t.strip() for r_t in tmp_list]

            additional_tags = []
            if key_info.split()[0] == 'X':
                # assert rows[8].value != None
                if rows[8].value != None:
                    additional_tags.append(rows[8].value)
                if rows[9].value != None:
                    additional_tags.append(rows[9].value)
                if rows[10].value != None:
                    additional_tags.append(rows[10].value)
            if rows[11].value != None:
                comment = rows[11].value
            else:
                comment = ""   
                
            tag = {}
            tag['query_id'] = query_name
            tag['answer_sent_index'] = answer_id['text']
            tag['linking'] = rows[4].value
            tag['answer_sent_text'] = prev_answer_text
            tag['key_tag'] = key_info
            tag['evidence_sent_text'] = evidence_sent_text
            tag['linking_speaker'] = linking_speaker
            tag['linking_text'] = linking_text
            tag['related_part'] = related_part
            tag['additional_tags'] = additional_tags
            tag['comment'] = comment
            tag['file_name'] = file_name
            tag['data_type'] = data_type 
            
            answers['sent'][answer_id['sentence-index']].append(tag)

        answers['data_type'] = data_type
        all_answers[query_name] = answers

    return


def get_acl2018_abssumm_data(file_name):
    """get acl2018 abssumm data"""
    dialog_act_based_transcripts = []
    prev_speaker_symbol = ""
    prev_speaker_name = ""

    f = codecs.open(os.path.join(get_project_dir(), "data/acl2018_abssumm_data/{}.da".format(file_name)))
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split("\t")

            dialog_act = parts[0]
            speaker_symbol = parts[3]
            speaker_name = parts[4]
            utterance = parts[-1]

            if prev_speaker_symbol != speaker_symbol:
                utterances = [{'text':utterance , 'dialog-act':dialog_act}]
                entry = {'speaker_symbol':speaker_symbol, 'speaker_name':speaker_name, 'utterances':utterances}
                dialog_act_based_transcripts.append(entry)
                prev_speaker_symbol = speaker_symbol
                prev_speaker_name = speaker_name
            else:
                assert prev_speaker_name == speaker_name
                dialog_act_based_transcripts[-1]['utterances'].append({'text':utterance , 'dialog-act':dialog_act})

    return dialog_act_based_transcripts


def mapping_transcripts_qmsum_sent_and_dialog_act_based(qmsum_data, dialog_act_based_transcripts):
    """mapping transcripts"""
    if len(qmsum_data['meeting_transcripts']) == len(dialog_act_based_transcripts):
        for qm_turn, da_turn in zip(qmsum_data['meeting_transcripts'], dialog_act_based_transcripts):
            turn_index = qm_turn['turn_index']
            qm_turn['content_da'] = []
            da_content = " ".join(u['text'] for u in da_turn['utterances'])
            da_content_replace_0 = " ".join(u['text'] if u['text'] != '0' else '' for u in da_turn['utterances'])

            if da_content == qm_turn['content']:
                for u_i, u in enumerate(da_turn['utterances']):
                    e = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text']}
                    qm_turn['content_da'].append(e)
            elif da_content_replace_0 == qm_turn['content']:
                for u_i, u in enumerate(da_turn['utterances']):
                    u['text'] = u['text'] if u['text'] != '0' else ''
                    e = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text']}
                    qm_turn['content_da'].append(e)

            else:
                print("da_content != qm_turn['content']")
                score = rouge(da_content, qm_turn['content'])
                if score > 0.8:
                    u_i = 0
                    for u in da_turn['utterances']:
                        if -1 ==  qm_turn['content'].find(u['text']):
                            continue
                        else:
                            e = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text']}
                            qm_turn['content_da'].append(e)
                        u_i += 1
                else:
                    print()
                    assert False
    else:
        i = 0
        for t_i, qm_turn in enumerate(qmsum_data['meeting_transcripts']):
            turn_index = qm_turn['turn_index']
            qm_turn['content_da'] = []

            da_turn = dialog_act_based_transcripts[i]
            da_content = " ".join(u['text'] for u in da_turn['utterances'])
            da_content_replace_0 = " ".join(u['text'] if u['text'] != '0' else '' for u in da_turn['utterances'])
            if da_content == qm_turn['content']:
                for u_i, u in enumerate(da_turn['utterances']):
                    e = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text']}
                    qm_turn['content_da'].append(e)
                i += 1
            elif da_content_replace_0 == qm_turn['content']:
                for u_i, u in enumerate(da_turn['utterances']):
                    u['text'] = u['text'] if u['text'] != '0' else ''
                    e = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text']}
                    qm_turn['content_da'].append(e)

                i += 1

    return qmsum_data



def merge_answers(answer_sent_based, speakers, answer_text):
    """merge answers"""
    new_answer_sent_based = copy.deepcopy(answer_sent_based)
    while True:
        merge_pairs = []
        for s_i, sent in enumerate(new_answer_sent_based[:-1]):
            for name, name_words in speakers.items():
                for w_i, n_word in enumerate(name_words[:-1]):
                    if sent['sent_words'][-1] == n_word: 
                        if new_answer_sent_based[s_i+1]['sent_words'][0] == name_words[w_i+1]:
                            if [s_i, s_i+1] not in merge_pairs:
                                merge_pairs.append([s_i, s_i+1])
                            break
                if len(merge_pairs) != 0: break

            if sent['sent_words'][-1] == 'i.e.' or sent['sent_words'][-1] == 'e.g.':
                if [s_i, s_i+1] not in merge_pairs:
                    merge_pairs.append([s_i, s_i+1])
                break
            elif sent['sent_words'][-1].find('i.e.') != -1 or sent['sent_words'][-1].find('e.g.') != -1:
                # print(sent)
                if [s_i, s_i+1] not in merge_pairs:
                    merge_pairs.append([s_i, s_i+1])
                break

            if len(merge_pairs) != 0: break

        if len(merge_pairs) == 0:
            break
        else:
            new2_answer_sent_based = []
            for pair in merge_pairs:
                start = pair[0]
                end = pair[1]
                assert start+1 == end
                for s_i, sent in enumerate(new_answer_sent_based):
                    if s_i == start:
                        sent['sent_i'] = sent['sent_i'] + "_" + new_answer_sent_based[s_i+1]['sent_i']
                        sent['sent_text'] = sent['sent_text'].strip() 
                        sent['sent_text'] = sent['sent_text'] + " " + new_answer_sent_based[s_i+1]['sent_text'].strip()
                        sent['sent_words'].extend(new_answer_sent_based[s_i+1]['sent_words'])

                        new2_answer_sent_based.append(sent)
                    elif s_i == end:
                        continue
                    else:
                        new2_answer_sent_based.append(sent)
        new_answer_sent_based = new2_answer_sent_based


    merged_answer = []

    for s_i, sent in enumerate(new_answer_sent_based):
        new_index = sent['index'] + "{}>".format(s_i)
        new_text = sent['sent_text'].strip()

        assert len(sent['sent_words']) == len(sent['sent_text'].split())
        merged_answer.append({'index':new_index, 'sent_text':new_text, 'merge_info':sent['sent_i']})

    merged_answer_text = ""
    for answer in merged_answer:
        merged_answer_text = merged_answer_text + " " + answer['sent_text']
        merged_answer_text = merged_answer_text.strip()

    answer_text = " ".join(answer_text.split())
    merged_answer_text = " ".join(merged_answer_text.split())
    assert "".join(merged_answer_text.split()) == "".join(answer_text.split())

    return merged_answer

def split_query_answers_using_sent_tokenizer(qmsum_data, speakers):
    """split query answers"""
    for g_i, general_query in enumerate(qmsum_data['general_query_list']):
        answer_sent_based = []
        for ii, sent in enumerate(sent_tokenize(general_query['answer'])):
            if len(qmsum_data['general_query_list']) == 1:
                answer_index = "<GB-"
            else:
                answer_index = "<GB{}-".format(g_i)
            e = {'index': answer_index, 'sent_i':str(ii), 'sent_text' : sent, 'sent_words': sent.split()}
            answer_sent_based.append(e)
        general_query['answer_sent_based'] =  merge_answers(answer_sent_based, speakers, general_query['answer'])

    for s_i, specific_query in enumerate(qmsum_data['specific_query_list']):
        answer_sent_based = []
        for ii, sent in enumerate(sent_tokenize(specific_query['answer'])):
            e = {'index': "<S{}-".format(s_i), 'sent_i':str(ii), 'sent_text' : sent, 'sent_words': sent.split()}
            answer_sent_based.append(e)
        specific_query['answer_sent_based'] = merge_answers(answer_sent_based, speakers, specific_query['answer'])

    return qmsum_data


def get_qmsum_data(file_name):
    """get qmsum data"""
    qmsum_data_path = os.path.join(get_project_dir(), "data/QMSum/{}.json".format(file_name))
    with open(qmsum_data_path, 'r') as f:
        qmsum_data = json.load(f)

    speakers = {}
    for t_i, turn in enumerate(qmsum_data['meeting_transcripts']):
        turn['turn_index'] = t_i
        sentences = []
        s_i = 0
        for sent in sent_tokenize(turn['content']):
            if sent.replace('.',"").strip() == "":
                continue
            sentences.append({'turn_index': t_i,'sent_index':s_i, 'text':sent})
            s_i += 1
        turn['content_check_sent'] = sentences

        if turn['speaker'] not in speakers.keys():
            speakers[turn['speaker']] = turn['speaker'].split()

    qmsum_data = split_query_answers_using_sent_tokenizer(qmsum_data, speakers)
    return qmsum_data


def get_summary_sentences(path):
    """get summary sentences"""
    summaries = []
    doc = minidom.parse(path)
    sentences = doc.getElementsByTagName('sentence')
    i = 0
    for sentence in sentences:
        if sentence.firstChild.data:
            e = {'index':"<GA-{}>".format(i), 'sent_text':sentence.firstChild.data}
            e['origin_id'] = sentence.getAttribute('nite:id')
            summaries.append(e)
            i += 1

    return summaries

def read_linking_file(rootdir,file_name, data_type, a, extracted_ambiguous, extracted_merges):
    """read linking file"""

    file_id = file_name.split('.')[0]
    qmsum_data = get_qmsum_data(file_id)
    if data_type == 'product' or data_type == 'academic':
        name = "data/abstractive_summary/{}.abssumm.xml".format(file_name.split('.')[0])
        abssumm_path = os.path.join(get_project_dir(), name)
        abstractive_summary = get_summary_sentences(abssumm_path)
        qmsum_data['generalA_summary'] = abstractive_summary
        dialog_act_based_transcripts = get_acl2018_abssumm_data(file_id)
        qmsum_data = mapping_transcripts_qmsum_sent_and_dialog_act_based(qmsum_data, dialog_act_based_transcripts)


    file_path = os.path.join(rootdir, file_name)
    excel_data = load_workbook(file_path, data_only=True)



    if data_type == 'product' or data_type == 'academic':
        load_GeneralA(excel_data, file_name, qmsum_data, data_type, a, extracted_ambiguous, extracted_merges)


    load_GeneralB(excel_data, file_name, qmsum_data, data_type, a, extracted_ambiguous, extracted_merges)
    load_Specific(excel_data, file_name, qmsum_data, data_type, a, extracted_ambiguous, extracted_merges)
    return qmsum_data




def make_w_f(workbook):
    """make workbook format"""
    w_f = {}

    bold_center_black   = workbook.add_format({'bold': True, 'align': 'center', 'font_color': 'black'})
    bold_center_black.set_align('vcenter')
    bold_center_black.set_text_wrap()
    w_f['bold_center_black'] = bold_center_black

    center_black   = workbook.add_format({'align': 'center', 'font_color': 'black'})
    center_black.set_align('vcenter')
    center_black.set_text_wrap()
    w_f['center_black'] = center_black

    center_gray   = workbook.add_format({'align': 'center', 'font_color': 'gray'})
    center_gray.set_align('vcenter')
    center_gray.set_text_wrap()
    w_f['center_gray'] = center_gray


    black   = workbook.add_format({'font_color': 'black'})
    black.set_align('vcenter')
    black.set_text_wrap()
    w_f['black'] = black

    bold_green   = workbook.add_format({'bold': True, 'font_color': 'green'})
    bold_green.set_align('vcenter')
    bold_green.set_text_wrap()
    w_f['bold_green'] = bold_green

    bold_black   = workbook.add_format({'bold': True, 'font_color': 'black'})
    bold_black.set_align('vcenter')
    bold_black.set_text_wrap()
    w_f['bold_black'] = bold_black

    bold_black_underline   = workbook.add_format({'bold': True, 'font_color': 'black'})
    bold_black_underline.set_align('vcenter')
    bold_black_underline.set_text_wrap()
    bold_black_underline.set_underline()
    w_f['bold_black_underline'] = bold_black_underline

    black_underline   = workbook.add_format({'font_color': 'black'})
    black_underline.set_align('vcenter')
    black_underline.set_text_wrap()
    black_underline.set_underline()
    w_f['black_underline'] = black_underline

    merge_format_black = workbook.add_format({
        'border':  1,
        'align':    'center',
        'valign':   'vcenter',
        'font_color': 'black'
    })
    w_f['merge_format_black'] = merge_format_black




    center_bg_yellow   = workbook.add_format({'align': 'center', 'border':  1, 'bg_color': 'FFFF99'})
    center_bg_yellow.set_align('vcenter')
    center_bg_yellow.set_text_wrap()
    w_f['center_bg_yellow'] = center_bg_yellow

    center_bg_red   = workbook.add_format({'align': 'center', 'bg_color': 'F5D1C9'})
    center_bg_red.set_align('vcenter')
    center_bg_red.set_text_wrap()
    w_f['center_bg_red'] = center_bg_red

    center_bg_green   = workbook.add_format({'align': 'center', 'bg_color': 'CCF5C1'})
    center_bg_green.set_align('vcenter')
    center_bg_green.set_text_wrap()
    w_f['center_bg_green'] = center_bg_green

    center_bg_blue   = workbook.add_format({'align': 'center', 'bg_color': 'D0E4F5'})
    center_bg_blue.set_align('vcenter')
    center_bg_blue.set_text_wrap()
    w_f['center_bg_blue'] = center_bg_blue

    return w_f


def merge_tag_analysis(worksheet, w_f, extracted_merges, tag_dict, worksheet_type):
    """merge tag analysis"""
    worksheet.set_column(1, 1, 25)

    worksheet.set_column(6, 6, 70)
    worksheet.set_column(7, 7, 25)

    row_i = 1
    worksheet.write(row_i, 1, "File-Query ID", w_f['bold_center_black'])
    worksheet.write(row_i, 2, "Sent ID", w_f['bold_center_black'])
    worksheet.write(row_i, 3, "Tag", w_f['bold_center_black'])
    worksheet.write(row_i, 4, "Merge-Sent-ID", w_f['bold_center_black'])
    worksheet.write(row_i, 6, "Origin-Summary-Sent-Text", w_f['bold_center_black'])
    worksheet.write(row_i, 7, "Count\nO-1 / O-2 / △ / X", w_f['bold_center_black'])
    row_i += 1

    for k, v_list in extracted_merges.items():
        assert len(v_list) > 1
        if v_list[0]['data_type'] != worksheet_type:
            continue

        row_i += 1
        splits = tag_dict[k]['answer_info']['file_name'].split('.')
        new_file_name = "{}-{}-{}".format(splits[0], splits[2], k.split('.')[-1])
        worksheet.merge_range(row_i,1, row_i+len(v_list)-1 ,1, new_file_name, w_f['merge_format_black'])

        for v in v_list:
            worksheet.write(row_i, 2, v['answer_id']['text'], w_f['center_black'])
            worksheet.write(row_i, 3, v['tag'], w_f['center_black'])
            if v['tag'] == '<Merge>':
                count_0_exact = 0
                count_1_semantic = 0
                count_2_abstractive = 0
                count_3_additional = 0

                for sent_index, sent_info in tag_dict[k]['sent'][v['answer_id']['text']].items():
                    if sent_info['key_tag'].split()[0] == 'O' and sent_info['key_tag'].split()[1] == '(1.exact)':
                        count_0_exact += 1
                    elif sent_info['key_tag'].split()[0] == 'O' and sent_info['key_tag'].split()[1] == '(2.semantic)':
                        count_1_semantic += 1
                    elif sent_info['key_tag'].split()[0] == '△':
                        count_2_abstractive += 1
                    elif sent_info['key_tag'].split()[0] == 'X':
                        count_3_additional += 1

                c0 = count_0_exact
                c1 = count_1_semantic
                c2 = count_2_abstractive
                c3 = count_3_additional
                worksheet.write(row_i, 7, "{} / {} / {} / {}".format(c0, c1, c2, c3), w_f['center_black'])

            if v['value'] != None:
                worksheet.write(row_i, 4, v['value'], w_f['center_black'])
            elif v['value'] == None and v['tag'] == '<Merge>':
                worksheet.write(row_i, 4, "", w_f['center_bg_yellow'])

            text =  tag_dict[k]['answer_info']['sent'][v['answer_id']['sentence-index']]['sent_text']
            worksheet.write(row_i, 6, text, w_f['black'])
            text = tag_dict[k]['answer_info']['file_name'].split('.')[2][2:]
            worksheet.write(row_i, 11, text, w_f['center_black'])

            row_i += 1


def write_underline_words(concate_list, text_split, ref_lower_split, underline_format, non_underline_format):
    """write underline words"""
    for w_i, word in enumerate(text_split):
        if w_i == len(text_split) -1 and word == '.':
            concate_list.append(non_underline_format)
            concate_list.append(word + " ")
            break
        word_list = [',', 'and','the', 'a', 'of', 'that', 'they', 'it', 'to']

        if word.lower() in ref_lower_split and word.lower() not in word_list:
            concate_list.append(underline_format)
            concate_list.append(word + " ")
        else:
            concate_list.append(non_underline_format)
            concate_list.append(word + " ")



def target_tag_analysis(worksheet, w_f, target_tag, all_tags):
    """target tag analysis"""
    worksheet.set_column(1, 1, 30)
    worksheet.set_column(5, 6, 10)

    worksheet.set_column(8, 8, 70)
    worksheet.set_column(9, 9, 70)
    # worksheet.set_column(9, 9, 70)

    row_i = 1
    worksheet.write(row_i, 1, "File-Query ID", w_f['bold_center_black'])
    worksheet.write(row_i, 2, "Sent ID", w_f['bold_center_black'])
    worksheet.write(row_i, 3, "Linking", w_f['bold_center_black'])
    worksheet.write(row_i, 5, "Similarity\nscore", w_f['bold_center_black'])
    worksheet.write(row_i, 6, "Matching\nscore", w_f['bold_center_black'])
    worksheet.write(row_i, 8, "Evidence-Sent-Text", w_f['bold_center_black'])
    worksheet.write(row_i, 9, "Summary-Sent-Text", w_f['bold_center_black'])
    row_i += 2


    for k, v in target_tag.items():

        new_words = []
        for sent in sent_tokenize(v['answer_sent_text']):
            if sent[-1] == '.' and len(sent) != 1:
                sent = sent[:-1] + " ."
            elif sent[-1] == '?' and len(sent) != 1:
                sent = sent[:-1] + " ?"
            elif sent[-1] == '!' and len(sent) != 1:
                sent = sent[:-1] + " !"
            for word in word_tokenize(sent):
                if word[-1] == ',' and len(word) != 1:
                    word = word[:-1] + " ,"
                new_words.append(word)
        v['answer_sent_text'] = " ".join(new_words)

        new_words = []
        for sent in sent_tokenize(v['evidence_sent_text']):
            if sent[-1] == '.' and len(sent) != 1:
                sent = sent[:-1] + " ."
            elif sent[-1] == '?' and len(sent) != 1:
                sent = sent[:-1] + " ?"
            elif sent[-1] == '!' and len(sent) != 1:
                sent = sent[:-1] + " !"
            for word in word_tokenize(sent):
                if word[-1] == ','and len(word) != 1:
                    word = word[:-1] + " ,"
                new_words.append(word)
        v['evidence_sent_text'] = " ".join(new_words)


        answer_sent_text = v['answer_sent_text'].lower()
        evidence_sent_text = v['evidence_sent_text'].lower()

        output = rouge_with_pyrouge(preds=[evidence_sent_text], refs=[answer_sent_text])
        rouge1_F, rouge2_F, rougeL_F, rouge1_P, rouge2_P, rougeL_P = output

        v['matching_score'] = (rouge1_P*0.5 + rouge2_P*0.3 + rougeL_P*0.2)*100

        embeddings1 = sentence_transformers_model.encode([evidence_sent_text], convert_to_tensor=True)
        embeddings2 = sentence_transformers_model.encode([answer_sent_text], convert_to_tensor=True)


        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        v['similarity_score'] = cosine_scores[0][0].cpu().tolist() * 100

        s = k.split('.')
        all_tags[s[0]][s[1]][s[2]][s[3]]['matching_score'] = (rouge1_P*0.5 + rouge2_P*0.3 + rougeL_P*0.2)*100
        all_tags[s[0]][s[1]][s[2]][s[3]]['similarity_score'] = v['similarity_score']


    # sorted_target_tag = sorted(target_tag.items(), key=lambda x: x[1]['matching_score'], reverse=True)

    sorted_target_tag = sorted(target_tag.items(), key=lambda x: x[1]['similarity_score'], reverse=True)

    for sorted_item in sorted_target_tag:
        k = sorted_item[0]
        v = sorted_item[1]
    # for k, v in target_tag.items():
        # print(k)

        splits = v['file_name'].split('.')


        new_file_name = "{}-{}-{}".format(splits[0], splits[2][2:], v['query_id'].split('.')[-1])

        worksheet.write(row_i, 1, new_file_name, w_f['center_black'])
        worksheet.write(row_i, 2, v['answer_sent_index'], w_f['center_black'])
        worksheet.write(row_i, 3, v['linking'], w_f['center_black'])

        worksheet.write(row_i, 5, '{:.2f}'.format(v['similarity_score']), w_f['center_black'])
        worksheet.write(row_i, 6, '{:.2f}'.format(v['matching_score']), w_f['center_black'])


        # worksheet.write(row_i, 7, v['evidence_sent_text'], w_f['black'])

        answer_sent_text_lower_split = v['answer_sent_text'].lower().split()
        concate_evidence_text = []
        concate_evidence_text.append(w_f['black'])
        concate_evidence_text.append(v['evidence_sent_text'].split()[0] + " ")

        split1 = v['evidence_sent_text'].split()[1:]
        a = answer_sent_text_lower_split
        write_underline_words(concate_evidence_text, split1, a , w_f['bold_green'], w_f['black'])
        worksheet.write_rich_string(row_i, 8, *concate_evidence_text, w_f['black'])
        a = all_tags[v['query_id'].split('.')[0]][v['query_id'].split('.')[1]]
        b = a[v['answer_sent_index']][v['linking']]
        b['concate_evidence_text'] = concate_evidence_text

        evidence_sent_text_lower_split = v['evidence_sent_text'].lower().split()

        related_part_start_end = []
        for r_p in v['related_part']:
            new_words = []
            for sent in sent_tokenize(r_p):
                if sent[-1] == '.' and len(sent) != 1:
                    sent = sent[:-1] + " ."
                elif sent[-1] == '?' and len(sent) != 1:
                    sent = sent[:-1] + " ?"
                elif sent[-1] == '!' and len(sent) != 1:
                    sent = sent[:-1] + " !"
                for word in word_tokenize(sent):
                    if word[-1] == ',' and len(word) != 1:
                        word = word[:-1] + " ,"
                    new_words.append(word)
            r_p = " ".join(new_words)
            if r_p[-2:] == ' .':
                start_position = v['answer_sent_text'].lower().find(r_p[:-2].lower())
                if start_position != -1:
                    related_part_start_end.append([start_position, start_position+len(r_p[:-2])])
            else:
                start_position = v['answer_sent_text'].lower().find(r_p.lower())
                if start_position != -1:
                    related_part_start_end.append([start_position, start_position+len(r_p)])


        worksheet.write_rich_string(row_i, 9, *concate_answer_text, w_f['black'])
        a= all_tags[v['query_id'].split('.')[0]][v['query_id'].split('.')[1]]
        a[v['answer_sent_index']][v['linking']]['concate_answer_text'] = concate_answer_text

        worksheet.write(row_i, 11, v['file_name'].split('.')[2][2:], w_f['center_black'])


        row_i += 1


def find_workers_scores(worker_name, all_tags):
    """find workers scores"""
    similarity_scores = {'0_exact':[], '1_semantic':[], '2_abstractive':[], '3_additional':[]}
    matching_scores = {'0_exact':[], '1_semantic':[], '2_abstractive':[], '3_additional':[]}
    tag_counts = {'0_exact':0, '1_semantic':0, '2_abstractive':0, '3_additional':0}
    for script_id, queries in all_tags.items():
        for query_id, sents in queries.items():
            for sent_id, sent_linkings in sents.items():
                for linking_id, linking_v in sent_linkings.items():
                    if linking_v['file_name'].split('.')[2][2:] == worker_name:
                        if linking_v['key_tag'].split()[0] == 'O':
                            similarity_scores['0_exact'].append(linking_v['similarity_score'])
                            matching_scores['0_exact'].append(linking_v['matching_score'])
                            tag_counts['0_exact'] += 1
                        elif linking_v['key_tag'].split()[0] == '△':
                            similarity_scores['2_abstractive'].append(linking_v['similarity_score'])
                            matching_scores['2_abstractive'].append(linking_v['matching_score'])
                            tag_counts['2_abstractive'] += 1
                        elif linking_v['key_tag'].split()[0] == 'X':
                            similarity_scores['3_additional'].append(linking_v['similarity_score'])
                            matching_scores['3_additional'].append(linking_v['matching_score'])
                            tag_counts['3_additional'] += 1

    return similarity_scores, matching_scores, tag_counts

def make_statistics(worksheet, w_f, workers, all_tags):
    """make statistics"""
    worksheet.set_column(1, 3, 8)

    worksheet.set_column(6, 9, 13)
    worksheet.set_column(11, 13, 13)

    row_i = 1

    worksheet.write(row_i, 4, "Total", w_f['bold_center_black'])
    worksheet.write(row_i, 6, "O-1.exact", w_f['bold_center_black'])
    worksheet.write(row_i, 7, "O-2.semantic", w_f['bold_center_black'])
    worksheet.write(row_i, 8, "△.abstractive", w_f['bold_center_black'])
    worksheet.write(row_i, 9, "X.additional", w_f['bold_center_black'])

    worksheet.write(row_i, 11, "X-1.supportive", w_f['bold_center_black'])
    worksheet.write(row_i, 12, "X-2.connective", w_f['bold_center_black'])
    worksheet.write(row_i, 13, "X-3.preliminary", w_f['bold_center_black'])


    for worker_id, worker_v in workers.items():
        if worker_v['tag_total_0123'] != 0:
            row_i += 3
            if worker_id == 'worker_kim':
                worker_name = "(김)"
                worker_bg = w_f['center_bg_green']
            elif worker_id == 'worker_lee':
                worker_name = "(이)"
                worker_bg = w_f['center_bg_red']
            elif worker_id == 'worker_park':
                worker_name = "(박)"
                worker_bg = w_f['center_bg_blue']

            worksheet.merge_range(row_i,1, row_i+5, 1, worker_name, worker_bg)
            worksheet.write(row_i, 3, "#query", w_f['center_black'])
            worksheet.write(row_i, 4, worker_v['n_query'], w_f['center_black'])


            row_i += 1

            worksheet.write(row_i, 3, "#tag", w_f['center_black'])
            worksheet.write(row_i, 4, worker_v['tag_total_0123'], w_f['center_black'])
            worksheet.write(row_i, 6, worker_v['tag_0_exact'], w_f['center_black'])
            worksheet.write(row_i, 7, worker_v['tag_1_semantic'], w_f['center_black'])
            worksheet.write(row_i, 8, worker_v['tag_2_abstractive'], w_f['center_black'])
            worksheet.write(row_i, 9, worker_v['tag_3_additional'], w_f['center_black'])

            worksheet.write(row_i, 11, worker_v['tag_3_1_supportive'], w_f['center_black'])
            worksheet.write(row_i, 12, worker_v['tag_3_2_connective'], w_f['center_black'])
            worksheet.write(row_i, 13, worker_v['tag_3_3_preliminary'], w_f['center_black'])


            row_i += 2
            output = find_workers_scores(worker_name, all_tags)
            similarity_scores_worker, matching_scores_worker, tag_counts_worker = output

            all_similarity_scores = []
            all_similarity_scores.extend(similarity_scores_worker['0_exact'])
            all_similarity_scores.extend(similarity_scores_worker['1_semantic'])
            all_similarity_scores.extend(similarity_scores_worker['2_abstractive'])
            all_similarity_scores.extend(similarity_scores_worker['3_additional'])


            row_i += 1
            all_matching_scores = []
            all_matching_scores.extend(matching_scores_worker['0_exact'])
            all_matching_scores.extend(matching_scores_worker['1_semantic'])
            all_matching_scores.extend(matching_scores_worker['2_abstractive'])
            all_matching_scores.extend(matching_scores_worker['3_additional'])


def query_analysis(worksheet, w_f, target_tag):
    """query analysis"""
    data_dict = {}
    for script_id, queries in target_tag['all'].items():
        for query_id, sents in queries.items():
            for sent_id, sent_linkings in sents.items():
                new_name = "{}.{}.{}".format(script_id, query_id, sent_id)
                key_counts = {}
                key_counts['0_exact'] = 0
                key_counts['1_semantic'] = 0
                key_counts['2_abstractive'] = 0
                key_counts['3_additional'] = 0
                

                evidence_sents = []
                concate_evidence_sents = []
                concate_answer_sents = []

                linkings = []
                similarity_scores = []
                matching_scores = []

                for linking_id, linking_v in sent_linkings.items():

                    linkings.append(linking_v['linking'])
                    matching_scores.append(linking_v['matching_score'])
                    similarity_scores.append(linking_v['similarity_score'])
                    if linking_v['key_tag'].split()[0] == 'O':
                        key_counts['0_exact'] += 1
                    elif linking_v['key_tag'].split()[0] == '△':
                        key_counts['2_abstractive'] += 1
                    elif linking_v['key_tag'].split()[0] == 'X':
                        key_counts['3_additional'] += 1
                    worker = linking_v['file_name'].split('.')[2][2:]
                    evidence_sents.append(linking_v['evidence_sent_text'])
                    concate_evidence_sents.append(linking_v['concate_evidence_text'])

                    answer_sent_text = linking_v['answer_sent_text']
                    concate_answer_sents.append(linking_v['concate_answer_text'])

                if len(similarity_scores) != 0:
                    matching_mean_score = np.mean(matching_scores)
                    similarity_mean_score = np.mean(similarity_scores)
                else:
                    matching_mean_score = 0
                    similarity_mean_score = 0

                evidence_sent_all = ""
                for text in evidence_sents:
                    evidence_sent_all = evidence_sent_all + " " + text
                evidence_sent_all = evidence_sent_all.strip()

                embeddings1 = sentence_transformers_model.encode([evidence_sent_all], convert_to_tensor=True)
                embeddings2 = sentence_transformers_model.encode([answer_sent_text.lower()], convert_to_tensor=True)

                #Compute cosine-similarities
                cosine_scores = util.cos_sim(embeddings1, embeddings2)
                total_similarity_score = cosine_scores[0][0].cpu().tolist() * 100

                rouge = rouge_with_pyrouge(preds=[evidence_sent_all], refs=[answer_sent_text.lower()])
                rouge1_F, rouge2_F, rougeL_F, rouge1_P, rouge2_P, rougeL_P = rouge
                
                total_matching_score = (rouge1_F*0.5 + rouge2_F*0.3 + rougeL_F*0.2)*100


                max_index = similarity_scores.index(max(similarity_scores))
                max_evidence_score_sent = evidence_sents[max_index]
                max_concate_evidence_text =  concate_evidence_sents[max_index]
                concate_answer_text = concate_answer_sents[max_index]


                assert new_name not in data_dict

                scores_sum = total_similarity_score*0.5 + max(similarity_scores)*0.3 + max(matching_scores)*0.2



                data_dict[new_name] = {}
                data_dict[new_name]['query_id'] =  "{}.{}".format(script_id, query_id),
                data_dict[new_name]['sent_id'] = sent_id
                data_dict[new_name]['linkings'] = linkings
                data_dict[new_name]['matching_scores'] = matching_scores
                data_dict[new_name]['matching_max_score'] = max(matching_scores)
                data_dict[new_name]['matching_min_score'] = min(matching_scores)
                data_dict[new_name]['matching_mean_score'] = matching_mean_score
                data_dict[new_name]['similarity_scores'] = similarity_scores
                data_dict[new_name]['similarity_max_score'] = max(similarity_scores)
                data_dict[new_name]['similarity_min_score'] = min(similarity_scores)
                data_dict[new_name]['similarity_mean_score'] = similarity_mean_score
                data_dict[new_name]['total_similarity_score'] = total_similarity_score
                data_dict[new_name]['total_matching_score'] = total_matching_score
                data_dict[new_name]['key_counts'] = key_counts
                data_dict[new_name]['worker'] = worker
                data_dict[new_name]['answer_sent_text'] = answer_sent_text
                data_dict[new_name]['max_evidence_score_sent'] = max_evidence_score_sent
                data_dict[new_name]['concate_answer_text'] = concate_answer_text
                data_dict[new_name]['max_concate_evidence_text'] = max_concate_evidence_text
                

    worksheet.set_column(1, 1, 30)
    worksheet.set_column(2, 2, 10)
    worksheet.set_column(3, 3, 20)
    worksheet.set_column(4, 4, 5)
    worksheet.set_column(5, 8, 5)

    worksheet.set_column(9, 9, 5)
    worksheet.set_column(11, 11, 5)

    worksheet.set_column(12, 13, 40)

    # worksheet.set_column(27, 28, 40)

    row_i = 0
    worksheet.merge_range(row_i,5, row_i, 8, "Counts - Tags", w_f['bold_center_black'])
    worksheet.merge_range(row_i,10, row_i+1, 10, "Ambiguousness", w_f['bold_center_black'])
    worksheet.merge_range(row_i,15, row_i, 18, "Similarity scores", w_f['bold_center_black'])
    worksheet.merge_range(row_i,20, row_i, 23, "Matching scores", w_f['bold_center_black'])


    row_i += 1
    worksheet.write(row_i, 1, "File-Query ID", w_f['bold_center_black'])
    worksheet.write(row_i, 2, "Sent ID", w_f['bold_center_black'])
    worksheet.write(row_i, 3, "Linkings", w_f['bold_center_black'])

    worksheet.write(row_i, 5, "O-1", w_f['bold_center_black'])
    worksheet.write(row_i, 6, "O-2", w_f['bold_center_black'])
    worksheet.write(row_i, 7, "△", w_f['bold_center_black'])
    worksheet.write(row_i, 8, "X", w_f['bold_center_black'])


    worksheet.write(row_i, 12, "Summary-Sent-Text", w_f['bold_center_black'])
    worksheet.write(row_i, 13, "Best Evidence-Sent-Text", w_f['bold_center_black'])


    worksheet.write(row_i, 15, "total", w_f['bold_center_black'])
    worksheet.write(row_i, 16, "max", w_f['bold_center_black'])
    worksheet.write(row_i, 17, "min", w_f['bold_center_black'])
    worksheet.write(row_i, 18, "avg", w_f['bold_center_black'])

    worksheet.write(row_i, 20, "total", w_f['bold_center_black'])
    worksheet.write(row_i, 21, "max", w_f['bold_center_black'])
    worksheet.write(row_i, 22, "min", w_f['bold_center_black'])
    worksheet.write(row_i, 23, "avg", w_f['bold_center_black'])


    row_i += 2


    sorted_data = sorted(data_dict.items(), key=lambda x: x[1]['total_similarity_score'])
    # sorted_data = sorted(sorted_data, key=lambda x: x[1]['total_similarity_score'])
    sorted_data = sorted(sorted_data, key=lambda x: x[1]['ambiguousness'], reverse=True)

    for sorted_item in sorted_data:
        k = sorted_item[0]
        v = sorted_item[1]

    # for k, v in data_dict.items():
    #     worksheet.write(row_i, 1, v['query_id'], w_f['center_black'])
        a1 = v['query_id'].split('.')[0]
        a2 = v['query_id'].split('.')[1]
        worksheet.write(row_i, 1, "{}-{}-{}".format(a1, v['worker'], a2), w_f['center_black'])

        worksheet.write(row_i, 2, v['sent_id'], w_f['center_black'])
        worksheet.write(row_i, 3, ", ".join(v['linkings']), w_f['black'])

        worksheet.write(row_i, 5, v['key_counts']['0_exact'], w_f['center_black'])
        worksheet.write(row_i, 6, v['key_counts']['1_semantic'], w_f['center_black'])
        worksheet.write(row_i, 7, v['key_counts']['2_abstractive'], w_f['center_black'])
        worksheet.write(row_i, 8, v['key_counts']['3_additional'], w_f['center_black'])


        worksheet.write(row_i, 10, '{:.2f}'.format(v['ambiguousness']), w_f['center_gray'])
        # worksheet.write(row_i, 27, '{:.2f}'.format(v['ambiguousness']), w_f['center_black'])


        worksheet.write_rich_string(row_i, 12, *v['concate_answer_text'], w_f['black'])
        worksheet.write_rich_string(row_i, 13, *v['max_concate_evidence_text'], w_f['black'])

        # worksheet.write(row_i, 27, v['answer_sent_text'], w_f['black'])
        # worksheet.write(row_i, 28, v['max_evidence_score_sent'], w_f['black'])


        worksheet.write(row_i, 15, '{:.2f}'.format(v['total_similarity_score']), w_f['center_black'])
        worksheet.write(row_i, 16, '{:.2f}'.format(v['similarity_max_score']), w_f['center_gray'])
        worksheet.write(row_i, 17, '{:.2f}'.format(v['similarity_min_score']), w_f['center_gray'])
        worksheet.write(row_i, 18, '{:.2f}'.format(v['similarity_mean_score']), w_f['center_gray'])


        worksheet.write(row_i, 20, '{:.2f}'.format(v['total_matching_score']), w_f['center_black'])
        worksheet.write(row_i, 21, '{:.2f}'.format(v['matching_max_score']), w_f['center_gray'])
        worksheet.write(row_i, 22, '{:.2f}'.format(v['matching_min_score']), w_f['center_gray'])
        worksheet.write(row_i, 23, '{:.2f}'.format(v['matching_mean_score']), w_f['center_gray'])

        worksheet.write(row_i, 25, v['worker'], w_f['center_black'])


        row_i += 1




def make_tags(extracted_answers, extracted_merges):
    """make tags"""
    tags = {}
    tags['product'] = {}
    tags['academic'] = {}
    tags['committee'] = {}
    
    tags['product']['tag_0_exact'] = {}
    tags['product']['tag_1_semantic'] = {}
    tags['product']['tag_2_abstractive'] = {}
    tags['product']['tag_3_additional'] = {}
    tags['product']['tag_3_1_supportive'] = {}
    tags['product']['tag_3_2_connective'] = {}
    tags['product']['tag_3_3_preliminary'] = {}
    tags['product']['all'] = {}
    
    tags['academic']['tag_0_exact'] = {}
    tags['academic']['tag_1_semantic'] = {}
    tags['academic']['tag_2_abstractive'] = {}
    tags['academic']['tag_3_additional'] = {}
    tags['academic']['tag_3_1_supportive'] = {}
    tags['academic']['tag_3_2_connective'] = {}
    tags['academic']['tag_3_3_preliminary'] = {}
    tags['academic']['all'] = {}
    
    tags['committee']['tag_0_exact'] = {}
    tags['committee']['tag_1_semantic'] = {}
    tags['committee']['tag_2_abstractive'] = {}
    tags['committee']['tag_3_additional'] = {}
    tags['committee']['tag_3_1_supportive'] = {}
    tags['committee']['tag_3_2_connective'] = {}
    tags['committee']['tag_3_3_preliminary'] = {}
    tags['committee']['all'] = {}
    

    product = {}
    
    product['worker_kim'] = {}
    product['worker_lee'] = {}
    product['worker_park'] = {}
    
    product['worker_kim']['tag_0_exact'] = 0
    product['worker_kim']['tag_1_semantic'] = 0
    product['worker_kim']['tag_2_abstractive'] = 0
    product['worker_kim']['tag_3_additional'] = 0
    product['worker_kim']['tag_3_1_supportive'] = 0
    product['worker_kim']['tag_3_2_connective'] = 0
    product['worker_kim']['tag_3_3_preliminary'] = 0
    product['worker_kim']['tag_total_0123'] = 0
    product['worker_kim']['n_query'] = 0

    product['worker_lee']['tag_0_exact'] = 0
    product['worker_lee']['tag_1_semantic'] = 0
    product['worker_lee']['tag_2_abstractive'] = 0
    product['worker_lee']['tag_3_additional'] = 0
    product['worker_lee']['tag_3_1_supportive'] = 0
    product['worker_lee']['tag_3_2_connective'] = 0
    product['worker_lee']['tag_3_3_preliminary'] = 0
    product['worker_lee']['tag_total_0123'] = 0
    product['worker_lee']['n_query'] = 0
    
    product['worker_park']['tag_0_exact'] = 0
    product['worker_park']['tag_1_semantic'] = 0
    product['worker_park']['tag_2_abstractive'] = 0
    product['worker_park']['tag_3_additional'] = 0
    product['worker_park']['tag_3_1_supportive'] = 0
    product['worker_park']['tag_3_2_connective'] = 0
    product['worker_park']['tag_3_3_preliminary'] = 0
    product['worker_park']['tag_total_0123'] = 0
    product['worker_park']['n_query'] = 0
        
    academic = {}
    
    academic['worker_kim'] = {}
    academic['worker_lee'] = {}
    academic['worker_park'] = {}
    
    academic['worker_kim']['tag_0_exact'] = 0
    academic['worker_kim']['tag_1_semantic'] = 0
    academic['worker_kim']['tag_2_abstractive'] = 0
    academic['worker_kim']['tag_3_additional'] = 0
    academic['worker_kim']['tag_3_1_supportive'] = 0
    academic['worker_kim']['tag_3_2_connective'] = 0
    academic['worker_kim']['tag_3_3_preliminary'] = 0
    academic['worker_kim']['tag_total_0123'] = 0
    academic['worker_kim']['n_query'] = 0

    academic['worker_lee']['tag_0_exact'] = 0
    academic['worker_lee']['tag_1_semantic'] = 0
    academic['worker_lee']['tag_2_abstractive'] = 0
    academic['worker_lee']['tag_3_additional'] = 0
    academic['worker_lee']['tag_3_1_supportive'] = 0
    academic['worker_lee']['tag_3_2_connective'] = 0
    academic['worker_lee']['tag_3_3_preliminary'] = 0
    academic['worker_lee']['tag_total_0123'] = 0
    academic['worker_lee']['n_query'] = 0
    
    academic['worker_park']['tag_0_exact'] = 0
    academic['worker_park']['tag_1_semantic'] = 0
    academic['worker_park']['tag_2_abstractive'] = 0
    academic['worker_park']['tag_3_additional'] = 0
    academic['worker_park']['tag_3_1_supportive'] = 0
    academic['worker_park']['tag_3_2_connective'] = 0
    academic['worker_park']['tag_3_3_preliminary'] = 0
    academic['worker_park']['tag_total_0123'] = 0
    academic['worker_park']['n_query'] = 0
        
        
    committee = {}
    
    committee['worker_kim'] = {}
    committee['worker_lee'] = {}
    committee['worker_park'] = {}
    
    committee['worker_kim']['tag_0_exact'] = 0
    committee['worker_kim']['tag_1_semantic'] = 0
    committee['worker_kim']['tag_2_abstractive'] = 0
    committee['worker_kim']['tag_3_additional'] = 0
    committee['worker_kim']['tag_3_1_supportive'] = 0
    committee['worker_kim']['tag_3_2_connective'] = 0
    committee['worker_kim']['tag_3_3_preliminary'] = 0
    committee['worker_kim']['tag_total_0123'] = 0
    committee['worker_kim']['n_query'] = 0

    committee['worker_lee']['tag_0_exact'] = 0
    committee['worker_lee']['tag_1_semantic'] = 0
    committee['worker_lee']['tag_2_abstractive'] = 0
    committee['worker_lee']['tag_3_additional'] = 0
    committee['worker_lee']['tag_3_1_supportive'] = 0
    committee['worker_lee']['tag_3_2_connective'] = 0
    committee['worker_lee']['tag_3_3_preliminary'] = 0
    committee['worker_lee']['tag_total_0123'] = 0
    committee['worker_lee']['n_query'] = 0
    
    committee['worker_park']['tag_0_exact'] = 0
    committee['worker_park']['tag_1_semantic'] = 0
    committee['worker_park']['tag_2_abstractive'] = 0
    committee['worker_park']['tag_3_additional'] = 0
    committee['worker_park']['tag_3_1_supportive'] = 0
    committee['worker_park']['tag_3_2_connective'] = 0
    committee['worker_park']['tag_3_3_preliminary'] = 0
    committee['worker_park']['tag_total_0123'] = 0
    committee['worker_park']['n_query'] = 0
        

    workers = {'product':product, 'academic':academic, 'committee':committee}

    tag_dict = {}

    for k, v_list in extracted_answers.items():
        # print(k)
        num_tag_per_query = {}
        num_tag_per_query['tag_0_exact'] = 0
        num_tag_per_query['tag_1_semantic'] = 0
        num_tag_per_query['tag_2_abstractive'] = 0
        num_tag_per_query['tag_3_additional'] = 0
        num_tag_per_query['tag_3_1_supportive'] = 0
        num_tag_per_query['tag_3_2_connective'] = 0
        num_tag_per_query['tag_3_3_preliminary'] = 0
        

        assert k not in tag_dict
        tag_dict[k] = {'answer_info': v_list['answer_info'], 'sent':{}}
        assert k not in tags[v_list['data_type']]['all']

        if k.split('.')[0] not in tags[v_list['data_type']]['all']:
            tags[v_list['data_type']]['all'][k.split('.')[0]] = {}
        tags[v_list['data_type']]['all'][k.split('.')[0]][k.split('.')[1]] = {}


        for i, vv_list in v_list['sent'].items():
            if len(vv_list) == 0:
                # assert k in extracted_merges
                if k in extracted_merges:
                    check = False
                    for tmp in extracted_merges[k]:
                        if tmp['answer_id']['sentence-index'] == i:
                            assert tmp['tag'] == '<Delete>'
                            check = True
                    assert check == True
                    continue
                else:
                    print(k)  ########## Need Check!!!!!!!!!!!! No Tag!!! if k == 'covid_8.specific3':
                    continue

            assert vv_list[0]['answer_sent_index'] not in tag_dict[k]['sent']
            tag_dict[k]['sent'][vv_list[0]['answer_sent_index']] = {}
            a = tags[v_list['data_type']]['all'][k.split('.')[0]][k.split('.')[1]]
            a[vv_list[0]['answer_sent_index']] = {}
            for v in vv_list:
                assert i == int(v['answer_sent_index'].split('-')[1][:-1])

                a = tags[v_list['data_type']]['all'][k.split('.')[0]][k.split('.')[1]]
                a[v['answer_sent_index']][v['linking']] = v.copy()
                assert v['linking'] not in tag_dict[k]['sent'][v['answer_sent_index']]   
                tag_dict[k]['sent'][v['answer_sent_index']][v['linking']] = v.copy()

                full_name = "{}.{}.{}".format(v['query_id'], v['answer_sent_index'], v['linking'])
                if v['key_tag'].split()[0] == 'O':
                    if v['key_tag'].split()[1] == '(1.exact)':
                        tags[v_list['data_type']]['tag_0_exact'][full_name] = v.copy()
                        num_tag_per_query['tag_0_exact'] += 1
                    elif v['key_tag'].split()[1] == '(2.semantic)':
                        tags[v_list['data_type']]['tag_1_semantic'][full_name] = v.copy()
                        num_tag_per_query['tag_1_semantic'] += 1
                    else:
                        assert False
                elif v['key_tag'].split()[0] == '△':
                    tags[v_list['data_type']]['tag_2_abstractive'][full_name] = v.copy()
                    num_tag_per_query['tag_2_abstractive'] += 1
                elif v['key_tag'].split()[0] == 'X':
                    tags[v_list['data_type']]['tag_3_additional'][full_name] = v.copy()
                    num_tag_per_query['tag_3_additional'] += 1

                    # assert len(v['additional_tags']) >= 1 ############# Need Check!!!!!!!!!!!
                    for add_tag in v['additional_tags']:
                        if add_tag.split('\n')[0] ==  'X-1 (supportive)':
                            tags[v_list['data_type']]['tag_3_1_supportive'][full_name] = v.copy()
                            num_tag_per_query['tag_3_1_supportive'] += 1
                        elif add_tag.split('\n')[0] ==  'X-2 (connective)':
                            tags[v_list['data_type']]['tag_3_2_connective'][full_name] = v.copy()
                            num_tag_per_query['tag_3_2_connective'] += 1
                        elif add_tag.split('\n')[0] ==  'X-3 (preliminary)':
                            tags[v_list['data_type']]['tag_3_3_preliminary'][full_name] = v.copy()
                            num_tag_per_query['tag_3_3_preliminary'] += 1
                        else:
                            print(add_tag)
                            assert False
                else:
                    assert False



    return tags, workers, tag_dict


def extract_additional_tags_info(queries):
    """extract additional tags info"""
    additional_tags = {}

    for query_name, summary_sents in queries.items():
        sents = []
        for sent_id, sent_linkings in summary_sents.items():
            tmp_sent = []
            for linking_id, linking_v in sent_linkings.items():

                new_v = {}
                new_v['query_id'] = linking_v['query_id']
                new_v['answer_sent_index'] = linking_v['answer_sent_index']
                new_v['linking'] = linking_v['linking']
                new_v['turn_index'] = linking_v['linking'].split('-')[0]
                new_v['sent_index'] = linking_v['linking'].split('-')[1]


                new_v['key_tag'] = linking_v['key_tag']
                new_v['similarity_score'] = linking_v['similarity_score']
                new_v['matching_score'] = linking_v['matching_score']
                new_v['evidence_sent_text'] = linking_v['evidence_sent_text']

                new_v['related_part'] = linking_v['related_part']
                new_v['additional_tags'] = linking_v['additional_tags']


                tmp_sent.append(new_v)

            sents.append(tmp_sent)
        additional_tags[query_name] = sents

    return additional_tags

def save_json_file(output_path, json_file_name, tags, all_qmsum_data_with_tag):
    """save json file"""
    file_path = os.path.join(output_path, json_file_name)

    for domain_name, domain_tags in tags.items():
        for script_id, queries in domain_tags['all'].items():
            assert script_id in all_qmsum_data_with_tag
            additional_tags = extract_additional_tags_info(queries)
            all_qmsum_data_with_tag[script_id]['additional_tags'] = additional_tags

    with open(file_path, 'w') as f:
        json.dump(all_qmsum_data_with_tag, f, indent=2)
    print()

if __name__ == "__main__":

    rootdir="data/results/"

    analysis_file_name = 'analysis'
    json_file_name = 'train.json'


    file_names = []
    for subdir, dirs, files in os.walk(rootdir):
        for file_name in files:
            if file_name[-5:] != '.xlsm':
                continue
            if file_name.split('.')[1] != "summarization":
                continue
            file_names.append(file_name)

    ami_file_names = get_file_names("./data/list.ami")
    icsi_file_names = get_file_names("./data/list.icsi")

    extracted_a = {}
    extracted_amb = {}
    merges = {}


    all_qmsum_data_with_tag = {}
    file_names.sort()

    data_types = []
    for file_name in file_names:
        print(file_name)

        file_id = file_name.split('.')[0]
        if file_id in ami_file_names:
            data_type = 'product'
        elif file_id in icsi_file_names:
            data_type = 'academic'
        else:
            data_type = 'committee'

        data_types.append(data_type)
        qmsum_data = read_linking_file(rootdir, file_name, data_type, extracted_a, extracted_amb, merges)
        qmsum_data['domain'] = data_type
        all_qmsum_data_with_tag[file_id] = qmsum_data

    assert len(file_names) == len(data_types)
    tags, workers, tag_dict = make_tags(extracted_a, extracted_merges)
    save_json_file(rootdir, json_file_name, tags, all_qmsum_data_with_tag)
