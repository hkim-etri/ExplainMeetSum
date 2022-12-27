"""Module providingFunction printing python version."""
import codecs
from xml.dom import minidom
import os

import json
from nltk.tokenize import sent_tokenize
import xlsxwriter
from rouge import Rouge
from tqdm import tqdm
import copy


def get_project_dir():
    """get project dir"""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return current_dir

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


def get_id(text):
    """get text id"""
    assert text[:3] == 'id('
    assert text[-1] == ')'
    cleaned_text = text[3:-1]

    return cleaned_text


def get_summlinks(extractive_path):
    """get summlinks"""
    summlinks = []
    doc = minidom.parse(extractive_path)
    sentences = doc.getElementsByTagName('summlink')
    for sentence in sentences:
        extractive_abstractive = sentence.getElementsByTagName('nite:pointer')
        ex_abs_info = {'extractive':'', 'abstractive':''}
        for s in extractive_abstractive:
            if s.getAttribute('role') == 'extractive':
                ex_abs_info['extractive'] = s.getAttribute('href')
            elif s.getAttribute('role') == 'abstractive':
                ex_abs_info['abstractive'] = s.getAttribute('href')
            else:
                assert False

        assert ex_abs_info['extractive'] != ''
        assert ex_abs_info['abstractive'] != ''

        summlinks.append(ex_abs_info)

    for summlink in summlinks:
        summlink['extractive'] = get_id(summlink['extractive'].split('#')[1])
        summlink['abstractive'] = get_id(summlink['abstractive'].split('#')[1])

    new_summlinks = {}
    for link in summlinks:
        if link['abstractive'] not in new_summlinks.keys():
            new_summlinks[link['abstractive']] = []
        new_summlinks[link['abstractive']].append(link['extractive'])

    summlink = {}
    for k, v_list in new_summlinks.items():
        for v in v_list:
            if v not in summlink.keys():
                summlink[v] = []
            summlink[v].append(k)

    return new_summlinks, summlink




def get_summary_sentences(path):
    """get summary sentences"""
    summaries = []
    doc = minidom.parse(path)
    sentences = doc.getElementsByTagName('sentence')
    for sentence in sentences:
        if sentence.firstChild.data:
            summaries.append({'id': sentence.getAttribute('nite:id'), 'text':sentence.firstChild.data})
    return summaries


def make_xlsx_transcripts(file_name, summlink, d, data_type, check_data, a_sum, qmsum_data):
    """make xlsx transcripts"""
    if data_type == 'ami':
        output_path = os.path.join(get_project_dir(), "data/output/xlsx_output/{}/1_Product/".format(check_data))
    elif data_type == 'icsi':
        output_path = os.path.join(get_project_dir(), "data/output/xlsx_output/{}/2_Academic/".format(check_data))
    else:
        output_path = os.path.join(get_project_dir(), "data/output/xlsx_output/{}/3_Committee/".format(check_data))

    transcripts_file_path = os.path.join(output_path, "{}.transcript.{}.xlsx".format(file_name,check_data))

    workbook = xlsxwriter.Workbook(transcripts_file_path)
    worksheet = workbook.add_worksheet()
    worksheet2 = workbook.add_worksheet()

    worksheet2.set_column(1, 1, 10)
    worksheet2.set_column(2, 2, 70)
    worksheet2.set_column(4, 4, 10)
    worksheet2.set_column(5, 5, 100)


    bold_center_red   = workbook.add_format({'bold': True, 'align': 'center', 'font_color': 'red'})
    bold_center_blue   = workbook.add_format({'bold': True, 'align': 'center', 'font_color': 'blue'})

    wrap_vcenter = workbook.add_format()
    wrap_vcenter.set_align('vcenter')
    wrap_vcenter.set_text_wrap()

    bold_green   = workbook.add_format({'bold': True, 'font_color': 'green'})
    bold_green.set_align('vcenter')
    bold_green.set_text_wrap()

    center = workbook.add_format()
    center.set_align('center')
    center.set_align('vcenter')
    center.set_text_wrap()

    center_gray = workbook.add_format({'font_color': 'gray'})
    center_gray.set_align('center')
    center_gray.set_align('vcenter')
    center_gray.set_text_wrap()

    center_border = workbook.add_format({'border':  1})
    center_border.set_align('center')
    center_border.set_align('vcenter')
    center_border.set_text_wrap()


    wrap_gray = workbook.add_format({'font_color': 'gray'})
    wrap_gray.set_align('vcenter')
    wrap_gray.set_text_wrap()

    wrap_gray_center = workbook.add_format({'font_color': 'gray'})
    wrap_gray_center.set_align('center')
    wrap_gray_center.set_align('vcenter')
    wrap_gray_center.set_text_wrap()

    merge_format = workbook.add_format({
        'border':  1,
        'align':    'center',
        'valign':   'vcenter',
    })
    merge_format.set_text_wrap()

    merge_bold_big_format = workbook.add_format({
        'bold': True,
        'border':  1,
        'valign':   'vcenter',
    })
    merge_bold_big_format.set_font_size(13)
    merge_bold_big_format.set_text_wrap()

    gray_format = workbook.add_format({
        'font_color': 'gray',
        'valign':   'vcenter',
    })
    gray_format.set_text_wrap()

    center_black_lightyellow   = workbook.add_format({'bg_color': 'FFFFCC'})
    center_black_lightyellow.set_align('center')
    center_black_lightyellow.set_align('vcenter')
    center_black_lightyellow.set_text_wrap()
    center_black_lightyellow.set_num_format('@')

    center_bold_black_lightred   = workbook.add_format({'bold': True, 'bg_color': 'FDC2C2'})
    center_bold_black_lightred.set_align('center')
    center_bold_black_lightred.set_align('vcenter')
    center_bold_black_lightred.set_text_wrap()
    center_bold_black_lightred.set_num_format('@')


    worksheet.set_column(1, 1, 15)
    worksheet.set_column(2, 3, 10)
    worksheet.set_column(4, 4, 100)
    worksheet.set_column(10, 10, 30)


    worksheet.write(0, 1, "Transcript ID", bold_center_blue)
    worksheet.write(0, 2, file_name, bold_center_red)

    # worksheet.write(0, 1, file_name, bold_center_red)

    worksheet.write(9, 1, "Speaker", bold_center_blue)
    worksheet.write(9, 2, "Turn-Index", bold_center_blue)
    worksheet.write(9, 4, "Content", bold_center_blue)

    if data_type == 'ami' or data_type == 'icsi':
        worksheet.write(9, 3, "DA-Index", bold_center_blue)
        worksheet.write(9, 10, "Dialog-Act", center_gray)
    else:
        worksheet.write(9, 3, "Sent-Index", bold_center_blue)


    all_turns = []
    row = 10

    if data_type == 'ami' or data_type == 'icsi':
        content_utter = 'content_da'
        utter_index = 'da_index'
    else:
        content_utter = 'content_check_sent'
        utter_index = 'sent_index'

    for t_i, turn in enumerate(d['meeting_transcripts']):
        start_row = row

        if len(turn[content_utter]) == 0:
            print("{} : len(turn[content_utter]) == 0".format(file_name))
            continue
        for utter_i, utter in enumerate(turn[content_utter]):
            turn_da_id = "{}-{}".format(t_i, utter_i)

            worksheet.write(row, 1, turn['speaker'], center_border)
            worksheet.write(row, 2, utter['turn_index'], center_border)
            worksheet.write(row, 3, utter[utter_index], center)

            all_turns.append({'id':turn_da_id, 'text':"<{}> : {}".format(turn['speaker'], utter['text'])})

            if data_type == 'ami' or data_type == 'icsi':
                worksheet.write(row, 10, utter['dialog-act'], center_gray)
                if utter['dialog-act'] not in summlink.keys():
                    worksheet.write(row, 4, utter['text'], wrap_vcenter)
                else:
                    worksheet.write(row, 4, utter['text'], bold_green)

                    Genaral_A_links = summlink[utter['dialog-act']]
                    Genaral_A_name = []
                    for link in Genaral_A_links:
                        GA_index = "<GA-{}>".format(int(link.split('.')[-1])-1)
                        if GA_index not in Genaral_A_name:
                            Genaral_A_name.append(GA_index)
                    worksheet.write(row, 5, "\n".join(Genaral_A_name), wrap_gray_center)
            else:
                worksheet.write(row, 4, utter['text'], wrap_vcenter)


            row += 1

        if start_row != row-1:
            worksheet.merge_range(start_row,1, row-1,1, turn['speaker'], merge_format)
            worksheet.merge_range(start_row,2, row-1,2, turn['turn_index'], merge_format)
        row += 1

    summary_ids = {}
    worksheet2_row1= 0
    worksheet2_row2= 0
    if data_type == 'ami' or data_type == 'icsi':
        summary_ids['GeneralA'] = []

        for sent in a_sum:
            s_id = "<GA-{}>".format(int(sent['id'].split('.')[-1])-1)
            s_text = sent['text']
            summary_ids['GeneralA'].append(s_id)
            worksheet2.write(worksheet2_row2, 4, s_id)
            worksheet2.write(worksheet2_row2, 5, s_text, wrap_vcenter)
            worksheet2_row2 += 1

        worksheet2.write(worksheet2_row1, 1, "GeneralA")
        worksheet2.write(worksheet2_row1, 2, "Summarize the whole meeting. (long)", wrap_vcenter)
        worksheet2_row1 += 1


    for g_i, general_query in enumerate(qmsum_data['general_query_list']):
        if len(qmsum_data['general_query_list']) == 1:
            GeneralB_name = "GeneralB"
        else:
            GeneralB_name = "GeneralB{}".format(g_i)
        summary_ids[GeneralB_name] = []
        for s_text in general_query['answer_sent_based']:
            summary_ids[GeneralB_name].append(s_text['index'])
            worksheet2.write(worksheet2_row2, 4, s_text['index'])
            worksheet2.write(worksheet2_row2, 5, s_text['sent_text'], wrap_vcenter)
            worksheet2_row2 += 1


        worksheet2.write(worksheet2_row1, 1, GeneralB_name)
        worksheet2.write(worksheet2_row1, 2, general_query['query'], wrap_vcenter)
        worksheet2_row1 += 1

    for s_i, specific_query in enumerate(qmsum_data['specific_query_list']):
        specific_name = "Specific{}".format(s_i)
        summary_ids[specific_name] = []
        for s_text in specific_query['answer_sent_based']:
            summary_ids[specific_name].append(s_text['index'])
            worksheet2.write(worksheet2_row2, 4, s_text['index'])
            worksheet2.write(worksheet2_row2, 5, s_text['sent_text'], wrap_vcenter)
            worksheet2_row2 += 1

        worksheet2.write(worksheet2_row1, 1, specific_name)
        worksheet2.write(worksheet2_row1, 2, specific_query['query'], wrap_vcenter)
        worksheet2_row1 += 1

    worksheet2_col3 = 0
    max_count3 = 0
    for k, v_list in summary_ids.items():
        worksheet2_row3 = worksheet2_row2 + 5
        worksheet2.write(worksheet2_row3, worksheet2_col3, k, center)
        worksheet2_row3 += 1
        if len(v_list) > max_count3:
            max_count3 = len(v_list)
        for v in v_list:
            worksheet2.write(worksheet2_row3, worksheet2_col3, v, center)
            worksheet2_row3 += 1
        worksheet2_col3 += 1


    worksheet2_row4 = worksheet2_row2 + 5 + max_count3 + 5
    worksheet2_col4 = 0
    for turn in all_turns:
        worksheet2.write(worksheet2_row4, worksheet2_col4, turn['id'], center)
        worksheet2_col4 += 1

    worksheet.set_row(3, 60)
    worksheet.set_row(4, 40)

    worksheet.write(1, 1, "Query ID", merge_format)
    worksheet.write(1, 2, "Answer ID", merge_format)

    worksheet.merge_range(2,0, 3,0, "1.", merge_format)

    worksheet.write(2, 3, "1-1. Query", gray_format)
    worksheet.write(3, 3, "1-2. Answer", gray_format)

    worksheet.merge_range(2,1, 3,1, "", merge_format)

    worksheet.write(3,2, "", merge_format)

    a = "=IFERROR(VLOOKUP($B$3:$B$4,Sheet2!$B$1:$C${},2,0),\"\")".format(worksheet2_row1)
    worksheet.write_formula(2,4, a, gray_format)

    a = "=IFERROR(VLOOKUP($C$4:$C$4,Sheet2!$E$1:$F${},2,0),\"\")".format(worksheet2_row2)
    worksheet.write_formula(3,4, a, merge_bold_big_format)


    if data_type == 'ami' or data_type == 'icsi':
        worksheet.write(4, 0, "2.", center_gray)
        worksheet.write(4, 1, "GeneralA", center_gray)
        worksheet.write(4, 2, "", center_gray)

        worksheet.write(4,3, "2-2. Answer", gray_format)

    worksheet.merge_range(1,5, 1,9, "Highlight (e.g. 10-2)", merge_format)
    worksheet.write(2, 5, "", center_black_lightyellow)
    worksheet.write(3, 5, "", center_black_lightyellow)
    worksheet.write(4, 5, "", center_black_lightyellow)
    worksheet.write(2, 6, "", center_black_lightyellow)
    worksheet.write(3, 6, "", center_black_lightyellow)
    worksheet.write(4, 6, "", center_black_lightyellow)
    worksheet.write(2, 7, "", center_black_lightyellow)
    worksheet.write(3, 7, "", center_black_lightyellow)
    worksheet.write(4, 7, "", center_black_lightyellow)
    worksheet.write(2, 8, "", center_black_lightyellow)
    worksheet.write(3, 8, "", center_black_lightyellow)
    worksheet.write(4, 8, "", center_black_lightyellow)
    worksheet.write(2, 9, "", center_black_lightyellow)
    worksheet.write(3, 9, "", center_black_lightyellow)
    worksheet.write(4, 9, "", center_black_lightyellow)

    worksheet.freeze_panes(6, 0)


    workbook.close()
    return all_turns


def make_workbook_format(workbook):
    """make workbook format"""
    workbook_format = {}

    bold_center_black   = workbook.add_format({'bold': True, 'align': 'center', 'font_color': 'black'})
    bold_center_black.set_align('vcenter')
    bold_center_black.set_text_wrap()
    workbook_format['bold_center_black'] = bold_center_black

    center_black   = workbook.add_format({'align': 'center', 'font_color': 'black'})
    center_black.set_align('vcenter')
    center_black.set_text_wrap()
    workbook_format['center_black'] = center_black

    bold_center_blue   = workbook.add_format({'bold': True, 'align': 'center', 'font_color': 'blue'})
    bold_center_blue.set_align('vcenter')
    workbook_format['bold_center_blue'] = bold_center_blue

    a = {'bold': True, 'align': 'center', 'font_color': 'blue', 'border':  1}
    bold_center_blue_border   = workbook.add_format(a)
    bold_center_blue_border.set_align('vcenter')
    bold_center_blue_border.set_text_wrap()
    workbook_format['bold_center_blue_border'] = bold_center_blue_border

    a = {'bold': True, 'align': 'center', 'font_color': 'red', 'border':  1, 'bg_color': 'FFFF99'}
    bold_center_red_yellow   = workbook.add_format(a)
    bold_center_red_yellow.set_align('vcenter')
    bold_center_red_yellow.set_text_wrap()
    bold_center_red_yellow.set_num_format('@')
    workbook_format['bold_center_red_yellow'] = bold_center_red_yellow


    center_black_lightyellow   = workbook.add_format({'bg_color': 'FFFFCC'})
    center_black_lightyellow.set_align('center')
    center_black_lightyellow.set_align('vcenter')
    center_black_lightyellow.set_text_wrap()
    workbook_format['center_black_lightyellow'] = center_black_lightyellow

    a = {'font_color': 'red', 'border':  1, 'bg_color': 'FFFFCC'}
    center_red_lightyellow   = workbook.add_format(a)
    center_red_lightyellow.set_align('center')
    center_red_lightyellow.set_align('vcenter')
    center_red_lightyellow.set_text_wrap()
    center_red_lightyellow.set_num_format('@')
    workbook_format['center_red_lightyellow'] = center_red_lightyellow

    a = {'font_color': 'blue', 'border':  1, 'bg_color': 'FFFFCC'}
    center_blue_lightyellow   = workbook.add_format(a)
    center_blue_lightyellow.set_align('center')
    center_blue_lightyellow.set_align('vcenter')
    center_blue_lightyellow.set_text_wrap()
    center_blue_lightyellow.set_num_format('@')
    workbook_format['center_blue_lightyellow'] = center_blue_lightyellow

    a = {'font_color': 'green', 'border':  1, 'bg_color': 'FFFFCC'}
    center_green_lightyellow   = workbook.add_format(a)
    center_green_lightyellow.set_align('center')
    center_green_lightyellow.set_align('vcenter')
    center_green_lightyellow.set_text_wrap()
    center_green_lightyellow.set_num_format('@')
    workbook_format['center_green_lightyellow'] = center_green_lightyellow

    a = {'bold': True, 'align': 'center', 'font_color': 'gray', 'border':  1, 'bg_color': 'FFFF99'}
    bold_center_gray_yellow   = workbook.add_format(a)
    bold_center_gray_yellow.set_align('vcenter')
    workbook_format['bold_center_gray_yellow'] = bold_center_gray_yellow

    a = {'font_color': 'black', 'border':  1, 'bg_color': 'FFFF99'}
    center_black_yellow   = workbook.add_format(a)
    center_black_yellow.set_align('center')
    center_black_yellow.set_align('vcenter')
    center_black_yellow.set_text_wrap()
    center_black_yellow.set_num_format('@')
    workbook_format['center_black_yellow'] = center_black_yellow

    a = {'font_color': 'black', 'border':  1, 'bg_color': 'FDE1B4'}
    center_black_orange_border   = workbook.add_format(a)
    center_black_orange_border.set_align('center')
    center_black_orange_border.set_align('vcenter')
    center_black_orange_border.set_text_wrap()
    center_black_orange_border.set_num_format('@')
    workbook_format['center_black_orange_border'] = center_black_orange_border

    a = {'font_color': 'black', 'border':  1, 'bg_color': 'FDC2C2'}
    center_black_red_border   = workbook.add_format(a)
    center_black_red_border.set_align('center')
    center_black_red_border.set_align('vcenter')
    center_black_red_border.set_text_wrap()
    center_black_red_border.set_num_format('@')
    workbook_format['center_black_red_border'] = center_black_red_border

    a = {'bold': True, 'font_color': 'green', 'border':  1, 'bg_color': 'E0E0E0'}
    center_green_gray_bold_border   = workbook.add_format(a)
    center_green_gray_bold_border.set_align('center')
    center_green_gray_bold_border.set_align('vcenter')
    center_green_gray_bold_border.set_text_wrap()
    center_green_gray_bold_border.set_num_format('@')
    workbook_format['center_green_gray_bold_border'] = center_green_gray_bold_border

    a = {'bold': True, 'font_color': 'blue', 'bg_color': 'white'}
    center_bold_blue_white   = workbook.add_format(a)
    center_bold_blue_white.set_align('center')
    center_bold_blue_white.set_align('vcenter')
    center_bold_blue_white.set_text_wrap()
    center_bold_blue_white.set_num_format('@')
    workbook_format['center_bold_blue_white'] = center_bold_blue_white

 
    a = {'bold': True, 'font_color': 'blue', 'border':  1, 'bg_color': 'E0E0E0'}
    center_blue_gray_bold_border   = workbook.add_format(a)
    center_blue_gray_bold_border.set_align('center')
    center_blue_gray_bold_border.set_align('vcenter')
    center_blue_gray_bold_border.set_text_wrap()
    center_blue_gray_bold_border.set_num_format('@')
    workbook_format['center_blue_gray_bold_border'] = center_blue_gray_bold_border


    center_gray   = workbook.add_format({'font_color': 'gray'})
    workbook_format['center_gray'] = center_gray

    a = {'bold': True, 'font_color': 'green', 'border':  1, 'bg_color': 'white'}
    center_white_border   = workbook.add_format(a)
    center_white_border.set_align('center')
    center_white_border.set_align('vcenter')
    center_white_border.set_text_wrap()
    center_white_border.set_num_format('@')
    workbook_format['center_white_border'] = center_white_border
    
    a = {'font_color': 'green', 'border':  1, 'bg_color': 'E0E0E0'}
    center_green_gray_border   = workbook.add_format(a)
    center_green_gray_border.set_align('center')
    center_green_gray_border.set_align('vcenter')
    center_green_gray_border.set_text_wrap()
    center_green_gray_border.set_num_format('@')
    workbook_format['center_green_gray_border'] = center_green_gray_border


    default_wrap   = workbook.add_format()
    default_wrap.set_align('vcenter')
    default_wrap.set_text_wrap()
    workbook_format['default_wrap'] = default_wrap

    bold_red   = workbook.add_format({'bold': True,  'font_color': 'red'})
    bold_red.set_align('vcenter')
    workbook_format['bold_red'] = bold_red

    wrap_vcenter_bold_border = workbook.add_format({'bold': True, 'border':  1})
    wrap_vcenter_bold_border.set_align('vcenter')
    wrap_vcenter_bold_border.set_text_wrap()
    workbook_format['wrap_vcenter_bold_border'] = wrap_vcenter_bold_border

    wrap_center_vcenter = workbook.add_format()
    wrap_center_vcenter.set_align('vcenter')
    wrap_center_vcenter.set_align('center')
    wrap_center_vcenter.set_text_wrap()
    workbook_format['wrap_center_vcenter'] = wrap_center_vcenter

    wrap_vcenter = workbook.add_format()
    wrap_vcenter.set_align('vcenter')
    wrap_vcenter.set_text_wrap()
    workbook_format['wrap_vcenter'] = wrap_vcenter

    merge_center_format = workbook.add_format({
        'border':  1,
        'align':    'center',
        'valign':   'vcenter',
    })
    merge_center_format.set_text_wrap()
    workbook_format['merge_center_format'] = merge_center_format

    merge_default_format = workbook.add_format({
        'border':  1,
        'valign':   'vcenter',
    })
    merge_default_format.set_text_wrap()
    workbook_format['merge_default_format'] = merge_default_format

    merge_big_font_format = workbook.add_format({
        'border':  1,
        'valign':   'vcenter',
    })
    merge_big_font_format.set_font_size(14)
    merge_big_font_format.set_text_wrap()
    workbook_format['merge_big_font_format'] = merge_big_font_format

    a = {'align': 'center', 'border':  1, 'bg_color': 'FFFF99'}
    center_bg_yellow   = workbook.add_format(a)
    center_bg_yellow.set_align('vcenter')
    center_bg_yellow.set_text_wrap()
    workbook_format['center_bg_yellow'] = center_bg_yellow

    merge_format_blue = workbook.add_format({
        'bold': True,
        'border':  1,
        'align':    'center',
        'valign':   'vcenter',
        'font_color': 'blue'
    })
    workbook_format['merge_format_blue'] = merge_format_blue

    merge_format_gray = workbook.add_format({
        'bold': True,
        'border':  1,
        'align':    'center',
        'valign':   'vcenter',
        'font_color': 'gray'
    })
    workbook_format['merge_format_gray'] = merge_format_gray

    return workbook_format


def make_xlsx_c_part(workbook, worksheet, tmp_worksheet, answers, all_turns, n_slot, tmp_prev_row):
    """make xlsx common part"""
    workbook.add_vba_project('./vbaProject.bin')
    workbook_format = make_workbook_format(workbook)

    worksheet.set_row(0, 50)
    worksheet.set_row(2, 50)
    worksheet.set_row(3, 50)

    worksheet.set_column(1, 1, 15)
    worksheet.set_column(2, 2, 10)
    # worksheet.set_column(3, 3, 60)
    worksheet.set_column(3, 3, 50)

    worksheet.set_column(4, 4, 15)
    worksheet.set_column(5, 5, 23)

    # worksheet.set_column(6, 6, 75)
    worksheet.set_column(6, 6, 65)
    worksheet.set_column(7, 7, 30)
    # worksheet.set_column(8, 8, 30)
    # worksheet.set_column(9, 11, 18)
    worksheet.set_column(8, 10, 18)
    worksheet.set_column(11, 11, 30)

    ##### write tmp_worksheet #####
    tmp_worksheet.write(0 + tmp_prev_row, 2, "O (1.exact)")
    tmp_worksheet.write(0 + tmp_prev_row, 3, "O (2.semantic)")
    tmp_worksheet.write(0 + tmp_prev_row, 4, "△ (abstractive)")
    tmp_worksheet.write(0 + tmp_prev_row, 5, "X (additional)")

    tmp_worksheet.write(1 + tmp_prev_row, 5, "X-1 (supportive)\n 근거/예시 제시")
    tmp_worksheet.write(2 + tmp_prev_row, 5, "X-2 (connective)\n 근거간 연결성 관련")
    tmp_worksheet.write(3 + tmp_prev_row, 5, "X-3 (preliminary)\n 대화 화제/소재 보강")


    tmp_worksheet.write(5 + tmp_prev_row, 0, "Ambiguous\n(1.segment)")
    tmp_worksheet.write(5 + tmp_prev_row, 1, "Ambiguous\n(2.all)")
    tmp_worksheet.write(6 + tmp_prev_row, 0, " -")
    tmp_worksheet.write(6 + tmp_prev_row, 1, " -")

    tmp_worksheet_col= 2
    tmp_worksheet_end_row = 0
    for t in all_turns:
        tmp_worksheet.write(5 + tmp_prev_row, tmp_worksheet_col, t['id'])
        tmp_worksheet.write(6 + tmp_prev_row, tmp_worksheet_col, t['text'])

        if len(answers[0]) > 4:
            for ii in range(7,len(answers[0])+7,8):
                tmp_worksheet.write(ii + tmp_prev_row, tmp_worksheet_col, "O (1.exact)")
                tmp_worksheet.write(ii+1 + tmp_prev_row, tmp_worksheet_col, "O (2.semantic)")
                tmp_worksheet.write(ii+2 + tmp_prev_row, tmp_worksheet_col, "△ (abstractive)")
                tmp_worksheet.write(ii+3 + tmp_prev_row, tmp_worksheet_col, "X (additional)")
                tmp_worksheet_end_row = ii+3+4
        else:
            tmp_worksheet.write(7 + tmp_prev_row, tmp_worksheet_col, "O (1.exact)")
            tmp_worksheet.write(8 + tmp_prev_row, tmp_worksheet_col, "O (2.semantic)")
            tmp_worksheet.write(9 + tmp_prev_row, tmp_worksheet_col, "△ (abstractive)")
            tmp_worksheet.write(10 + tmp_prev_row, tmp_worksheet_col, "X (additional)")
            tmp_worksheet_end_row = 10
        tmp_worksheet_col += 1

    tmp_worksheet.write(5 + tmp_prev_row, tmp_worksheet_col, "<Merge>")
    tmp_worksheet.write(6 + tmp_prev_row, tmp_worksheet_col, " -")
    for a_i, a in enumerate(answers[0]):
        tmp_worksheet_row = a_i+5+2
        tmp_worksheet.write(tmp_worksheet_row + tmp_prev_row, tmp_worksheet_col, a)
    tmp_worksheet_col += 1
    tmp_worksheet.write(5 + tmp_prev_row, tmp_worksheet_col, "<Delete>")
    tmp_worksheet.write(6 + tmp_prev_row, tmp_worksheet_col, " -")

    assert tmp_worksheet_end_row >= tmp_worksheet_row
    ##### write tmp_worksheet #####


    worksheet.write(0, 1, "Transcript ID", workbook_format['bold_center_black'])
    worksheet.write(0, 3, file_name, workbook_format['bold_red'])
    worksheet.write(5, 1, "Answer", workbook_format['bold_center_blue'])


    linking = []
    linking.append(workbook_format['bold_center_red_yellow'])
    linking.append("Linking")
    linking.append(workbook_format['bold_center_gray_yellow'])
    linking.append("\n(e.g. 10-2)")
    worksheet.write_rich_string(4, 4, *linking, workbook_format['center_bg_yellow'])

    key = []
    key.append(workbook_format['bold_center_red_yellow'])
    key.append("Key")
    key.append(workbook_format['bold_center_gray_yellow'])
    key.append("\n( O,△,X )")
    worksheet.write_rich_string(4, 5, *key, workbook_format['center_bg_yellow'])



    worksheet.write(4, 6, "Evidence sentence", workbook_format['merge_default_format'])
    worksheet.write(4, 7, "Related Part", workbook_format['merge_default_format'])
    # worksheet.write(4, 8, "Comment", workbook_format['merge_default_format'])
    # worksheet.merge_range(4,9, 4,11, "Additional Tagging", workbook_format['merge_center_format'])
    worksheet.merge_range(4,8, 4,10, "Additional Tagging", workbook_format['merge_center_format'])
    worksheet.write(4, 11, "Comment", workbook_format['merge_default_format'])


    worksheet.write(0, 6, "", workbook_format['wrap_vcenter'])
    worksheet.write(1, 6, "", workbook_format['wrap_vcenter'])
    worksheet.write(2, 6, "", workbook_format['wrap_vcenter'])
    worksheet.write(3, 6, "", workbook_format['wrap_vcenter'])


    row = 5
    start_row = row
    for a_i, a in zip(answers[0], answers[1]):
        start_row2 = row

        a_n_word = len(a.split())
        if a_n_word > n_slot:
            final_n_slot = int(a_n_word * 1.1)
            if final_n_slot > n_slot * 1.5 :
                final_n_slot = int(n_slot*1.5)
        else:
            final_n_slot = n_slot


        for _ in range(final_n_slot):
            worksheet.write(row, 4, "", workbook_format['bold_center_red_yellow'])
            worksheet.write(row, 5, "", workbook_format['center_black_yellow'])

            a = "=IFERROR(HLOOKUP($E{},tmp!${}:${},2,0),\"\")".format(row+1, 6+tmp_prev_row, 7+tmp_prev_row)
            worksheet.write_formula(row, 6, a, workbook_format['wrap_vcenter'])

            worksheet.write(row, 7, "", workbook_format['wrap_vcenter'])
            worksheet.write(row, 11, "", workbook_format['wrap_vcenter'])

            row += 1

        worksheet.merge_range(start_row2,2, row-1,2, a_i, workbook_format['merge_center_format'])
        worksheet.merge_range(start_row2,3, row-1,3, a, workbook_format['merge_big_font_format'])

        row += 1

    if start_row != row-1:
        worksheet.merge_range(start_row,1, row-2,1, "Answer", workbook_format['merge_format_blue'])


    tmp_prev_row += tmp_worksheet_end_row
    tmp_prev_row += 10

    return worksheet, workbook_format, row, tmp_prev_row


def make_xlsx_GeneralA(workbook, w_l, summaries, qmsum_data, all_turns, tmp_prev_row):
    """make xlsx generalA query part"""
    answers_id = []
    answers_sent = []
    for ii, sent in enumerate(summaries):
        answers_id.append("<GA-{}>".format(int(sent['id'].split('.')[-1])-1))
        answers_sent.append(sent['text'])

    a = [answers_id, answers_sent]
    output = make_xlsx_c_part(workbook, w_l[0], w_l[3], a, all_turns, 20, tmp_prev_row)
    worksheet, workbook_format, row, tmp_prev_row = output

    worksheet.write(2, 1, "Query\n(GeneralA)", workbook_format['bold_center_blue_border'])
    worksheet.write(2, 3, "Summarize the whole meeting. (long)")

    row += 6
    worksheet.write(row-1, 4, "Relevant Span", workbook_format['wrap_center_vcenter'])
    start_row = row
    for t_i, topic in enumerate(qmsum_data['topic_list']):
        worksheet.write(row, 2, t_i, workbook_format['wrap_center_vcenter'])
        worksheet.write(row, 3, topic['topic'], workbook_format['wrap_vcenter'])
        new_span = "[ "
        for ii, span in enumerate(topic['relevant_text_span']):
            if ii != 0:
                new_span = new_span + ", "
            new_span = new_span + "({} , {})".format(int(span[0]), int(span[1]))
        new_span = new_span + " ]"
        worksheet.write(row, 4, new_span, workbook_format['wrap_vcenter'])
        row += 1

    if start_row != row-1:
        worksheet.merge_range(start_row,1, row-1,1, "Topics", workbook_format['merge_format_gray'])
    else:
        worksheet.write(start_row,1, "Topics", workbook_format['merge_format_gray'])


    return tmp_prev_row

def make_xlsx_GeneralB(workbook, w_l, qmsum_data, oracle_data, all_turns, data_type, tmp_prev_row):
    """make xlsx generalB query part"""

    for g_i, general_query in enumerate(qmsum_data['general_query_list']):

        answers_index = []
        answers_sent = []
        for ii, sent in enumerate(general_query['answer_sent_based']):
            if len(qmsum_data['general_query_list']) == 1:
                tmp_index = "<GB-{}>".format(ii)
            else:
                tmp_index = "<GB{}-{}>".format(g_i, ii)

            assert tmp_index == sent['index']
            answers_index.append(sent['index'])
            answers_sent.append(sent['sent_text'])

        a = [answers_index, answers_sent]
        output = make_xlsx_c_part(workbook, w_l[1][g_i], w_l[3], a, all_turns, 15, tmp_prev_row)
        worksheet, workbook_format, row, tmp_prev_row = output

        if len(qmsum_data['general_query_list']) == 1:
            worksheet.write(2, 1, "Query\n(GeneralB)", workbook_format['bold_center_blue_border'])
        else:
            worksheet.write(2, 1, "Query\n(GeneralB{})".format(g_i))
        worksheet.write(2, 3, general_query['query'], workbook_format['wrap_vcenter_bold_border'])

        row += 6
        worksheet.write(row-1, 4, "Relevant Index", workbook_format['wrap_center_vcenter'])
        start_row = row
        query_id = "general{}".format(g_i)
        for o_i, oracle_sent in enumerate(oracle_data[query_id]['oracle_info']):
            worksheet.write(row, 2, o_i, workbook_format['wrap_center_vcenter'])

            if data_type == 'ami' or data_type == 'icsi':
                a =  qmsum_data['meeting_transcripts'][oracle_sent['turn_i']]['content_da']
                text = a[oracle_sent['da_i']]['text']
            else:
                a = qmsum_data['meeting_transcripts'][oracle_sent['turn_i']]['content_check_sent']
                text = a[oracle_sent['sent_i']]['text']

            speaker = "<{}> : ".format(qmsum_data['meeting_transcripts'][oracle_sent['turn_i']]['speaker'])
            worksheet.write(row, 3, speaker + text, workbook_format['wrap_vcenter'])

            if data_type == 'ami' or data_type == 'icsi':
                worksheet.write(row, 4, "{}-{}".format(oracle_sent['turn_i'], oracle_sent['da_i']))
            else:
                worksheet.write(row, 4, "{}-{}".format(oracle_sent['turn_i'], oracle_sent['sent_i']))

            row += 1

        if start_row != row-1:
            worksheet.merge_range(start_row,1, row-1,1, "Oracle", workbook_format['merge_format_gray'])
        else:
            worksheet.write(start_row,1, "Oracle", workbook_format['merge_format_gray'])



        row += 6
        worksheet.write(row-1, 4, "Relevant Span", workbook_format['wrap_center_vcenter'])
        start_row = row
        for t_i, topic in enumerate(qmsum_data['topic_list']):
            worksheet.write(row, 2, t_i, workbook_format['wrap_center_vcenter'])
            worksheet.write(row, 3, topic['topic'], workbook_format['wrap_vcenter'])
            new_span = "[ "
            for ii, span in enumerate(topic['relevant_text_span']):
                if ii != 0:
                    new_span = new_span + ", "
                new_span = new_span + "({} , {})".format(int(span[0]), int(span[1]))
            new_span = new_span + " ]"
            worksheet.write(row, 4, new_span, workbook_format['wrap_vcenter'])
            row += 1

        if start_row != row-1:
            worksheet.merge_range(start_row,1, row-1,1, "Topics", workbook_format['merge_format_gray'])
        else:
            worksheet.write(start_row,1, "Topics", workbook_format['merge_format_gray'])


    return tmp_prev_row


def make_xlsx_Specific(workbook, w_l, qmsum_data, oracle_data, all_turns, data_type, tmp_prev_row):
    """make xlsx specific query part"""
    for s_i, specific_query in enumerate(qmsum_data['specific_query_list']):

        answers_index = []
        answers_sent = []
        for ii, sent in enumerate(specific_query['answer_sent_based']):
            tmp_index = "<S{}-{}>".format(s_i, ii)

            assert tmp_index == sent['index']
            answers_index.append(sent['index'])
            answers_sent.append(sent['sent_text'])

        a = [answers_index, answers_sent]
        output = make_xlsx_c_part(workbook, w_l[2][s_i], w_l[3], a, all_turns, 10, tmp_prev_row)
        worksheet, workbook_format, row, tmp_prev_row = output

        worksheet.write(2, 1, "Query\n(Specific{})".format(s_i), workbook_format['bold_center_blue_border'])
        worksheet.write(3, 1, "Relevant Span", workbook_format['bold_center_blue'])
        worksheet.write(2, 3, specific_query['query'], workbook_format['wrap_vcenter_bold_border'])

        new_span = "[ "
        for ii, span in enumerate(specific_query['relevant_text_span']):
            if ii != 0:
                new_span = new_span + ", "
            new_span = new_span + "({} , {})".format(int(span[0]), int(span[1]))
        new_span = new_span + " ]"
        worksheet.write(3, 3, new_span, workbook_format['wrap_vcenter'])

        row += 6
        worksheet.write(row-1, 4, "Relevant Index", workbook_format['wrap_center_vcenter'])
        start_row = row
        query_id = "specific{}".format(s_i)
        # print(query_id)
        for o_i, oracle_sent in enumerate(oracle_data[query_id]['oracle_info']):
            worksheet.write(row, 2, o_i, workbook_format['wrap_center_vcenter'])

            if data_type == 'ami' or data_type == 'icsi':
                utter_i = oracle_sent['da_i']
                content_utter = 'content_da'
            else:
                utter_i = oracle_sent['sent_i']
                content_utter = 'content_check_sent'

            text = qmsum_data['meeting_transcripts'][oracle_sent['turn_i']][content_utter][utter_i]['text']
            speaker = "<{}> : ".format(qmsum_data['meeting_transcripts'][oracle_sent['turn_i']]['speaker'])
            worksheet.write(row, 3, speaker + text, workbook_format['wrap_vcenter'])
            worksheet.write(row, 4, "{}-{}".format(oracle_sent['turn_i'], utter_i))

            row += 1

        if start_row != row-1:
            worksheet.merge_range(start_row,1, row-1,1, "Oracle", workbook_format['merge_format_gray'])
        else:
            worksheet.write(start_row,1, "Oracle", workbook_format['merge_format_gray'])


    return tmp_prev_row


def rouge(dec, ref):
    """make rouge scoure"""
    if dec == '' or ref == '':
        return 0.0
    rouge = Rouge()
    scores = rouge.get_scores(dec, ref)
    return (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3


def make_xlsx(file_name, a_sum, summlink, qmsum_data, data_type, check_data):
    """make xlsm"""
    output_path1 = os.path.join(get_project_dir(), "data/output/xlsx_output/{}".format(check_data))
    if not os.path.exists(output_path1):
        os.makedirs(output_path1)
    if data_type == 'ami':
        output_path2 = os.path.join(get_project_dir(), "data/output/xlsx_output/{}/1_Product".format(check_data))
    elif data_type == 'icsi':
        output_path2 = os.path.join(get_project_dir(), "data/output/xlsx_output/{}/2_Academic".format(check_data))
    else:
        output_path2 = os.path.join(get_project_dir(), "data/output/xlsx_output/{}/3_Committee".format(check_data))
    if not os.path.exists(output_path2):
        os.makedirs(output_path2)

    all_turns = make_xlsx_transcripts(file_name, summlink, qmsum_data, data_type, check_data, a_sum, qmsum_data)


    summarization_file_path = os.path.join(output_path2, "{}.summarization.{}.xlsm".format(file_name, check_data))
    sum_wb = xlsxwriter.Workbook(summarization_file_path)

    if data_type == 'ami' or data_type == 'icsi':
        generalA_worksheet = sum_wb.add_worksheet('generalA')
    generalB_w_l = []
    for g_i, _ in enumerate(qmsum_data['general_query_list']):
        if len(qmsum_data['general_query_list']) == 1:
            name = "generalB"
        else:
            name = "generalB{}".format(g_i)
        generalB_w_l.append(sum_wb.add_worksheet(name))
    specific_w_l = []
    for s_i, _ in enumerate(qmsum_data['specific_query_list']):
        name = "specific{}".format(s_i)
        specific_w_l.append(sum_wb.add_worksheet(name))
    tmp_worksheet = sum_wb.add_worksheet('tmp')

    if data_type == 'ami' or data_type == 'icsi':
        w_l = [generalA_worksheet, generalB_w_l, specific_w_l, tmp_worksheet]
    else:
        w_l = [None, generalB_w_l, specific_w_l, tmp_worksheet]


    tmp_prev_row = 0
    if data_type == 'ami' or data_type == 'icsi':
        tmp_prev_row = make_xlsx_GeneralA(sum_wb, w_l, a_sum, qmsum_data, all_turns, tmp_prev_row)

    oracle_path = os.path.join(get_project_dir(), "data/oracle-v0.3.2/{}.json".format(file_name))
    with open(oracle_path, 'r') as f:
        oracle_data = json.load(f)
    oracle_dict = {}
    for d in oracle_data:
        for oracle_sent in d['oracle_info']:
            if data_type == 'ami' or data_type == 'icsi':
                assert 'da_i' in oracle_sent.keys()
                assert 'sent_i' not in oracle_sent.keys()
            else:
                assert 'sent_i' in oracle_sent.keys()
                assert 'da_i' not in oracle_sent.keys()
        oracle_dict[d['query_id']] = d

    tmp_prev_row = make_xlsx_GeneralB(sum_wb, w_l, qmsum_data, oracle_dict, all_turns, data_type, tmp_prev_row)
    tmp_prev_row = make_xlsx_Specific(sum_wb, w_l, qmsum_data, oracle_dict, all_turns, data_type, tmp_prev_row)

    sum_wb.close()
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
                u = [{'text':utterance , 'dialog-act':dialog_act}]
                a = {'speaker_symbol':speaker_symbol, 'speaker_name':speaker_name, 'utterances':u}
                dialog_act_based_transcripts.append(a)
                prev_speaker_symbol = speaker_symbol
                prev_speaker_name = speaker_name
            else:
                assert prev_speaker_name == speaker_name
                dialog_act_based_transcripts[-1]['utterances'].append({'text':utterance , 'dialog-act':dialog_act})

    return dialog_act_based_transcripts



def merge_answers(answer_sent_based, speakers, answer_text):
    """merge answers"""
    new_answer_sent_based = copy.deepcopy(answer_sent_based)
    while True:
        merge_pairs = []
        for s_i, sent in enumerate(new_answer_sent_based[:-1]):
            for name, name_words in speakers.items():
                for w_i, n_word in enumerate(name_words[:-1]):
                    if sent['sent_words'][-1] == n_word :
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
                        sent['sent_text'] = sent['sent_text'] + " " + new_answer_sent_based[s_i+1]['sent_text']
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
            a = {'index': answer_index, 'sent_i':str(ii), 'sent_text' : sent, 'sent_words': sent.split()}
            answer_sent_based.append(a)
        general_query['answer_sent_based'] =  merge_answers(answer_sent_based, speakers, general_query['answer'])

    for s_i, specific_query in enumerate(qmsum_data['specific_query_list']):
        answer_sent_based = []
        for ii, sent in enumerate(sent_tokenize(specific_query['answer'])):
            a = {'index': "<S{}-".format(s_i), 'sent_i':str(ii), 'sent_text' : sent, 'sent_words': sent.split()}
            answer_sent_based.append(a)
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
                    a = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text'], 'dialog-act': u['dialog-act']}
                    qm_turn['content_da'].append(a)
            elif da_content_replace_0 == qm_turn['content']:
                for u_i, u in enumerate(da_turn['utterances']):
                    u['text'] = u['text'] if u['text'] != '0' else ''
                    a = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text'], 'dialog-act': u['dialog-act']}
                    qm_turn['content_da'].append(a)

            else:
                print("da_content != qm_turn['content']")
                score = rouge(da_content, qm_turn['content'])
                if score > 0.8:
                    u_i = 0
                    for u in da_turn['utterances']:
                        if -1 ==  qm_turn['content'].find(u['text']):
                            continue
                        else:
                            a = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text']}
                            qm_turn['content_da'].append(a)
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
                    a = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text']}
                    qm_turn['content_da'].append(a)
                i += 1
            elif da_content_replace_0 == qm_turn['content']:
                for u_i, u in enumerate(da_turn['utterances']):
                    u['text'] = u['text'] if u['text'] != '0' else ''
                    a = {'turn_index': turn_index, 'da_index':u_i, 'text': u['text']}
                    qm_turn['content_da'].append(a)

                i += 1


    return qmsum_data



def run(file_name, data_type, check_data):
    """run"""
    qmsum_data = get_qmsum_data(file_name)

    if data_type == 'ami' or data_type == 'icsi':
        abssumm_path = os.path.join(get_project_dir(), "data/abstractive_summary/{}.abssumm.xml".format(file_name))
        summlink_path = os.path.join(get_project_dir(), "data/extractive_summary/{}.summlink.xml".format(file_name))
        a_sum = get_summary_sentences(abssumm_path)
        summlinks, summlink = get_summlinks(summlink_path)
        dialog_act_based_transcripts = get_acl2018_abssumm_data(file_name)
        qmsum_data = mapping_transcripts_qmsum_sent_and_dialog_act_based(qmsum_data, dialog_act_based_transcripts)
    else:
        a_sum = []
        summlink = {}

    make_xlsx(file_name, a_sum, summlink, qmsum_data, data_type, check_data)

    return



if __name__ == "__main__":

    check_data = "v3"

    rootdir="/data/QMSum/data/ALL/all/"
    file_names = []
    for subdir, dirs, files in os.walk(rootdir):
        for file_name in files:
            if file_name[-5:] != '.json':
                continue
            file_names.append(file_name[:-5])

    ami_file_names = get_file_names("./data/list.ami")
    icsi_file_names = get_file_names("./data/list.icsi")

    for file_name in tqdm(file_names):

        if file_name in ami_file_names:
            data_type = 'ami'
        elif file_name in icsi_file_names:
            data_type = 'icsi'
        else:
            data_type = 'committee'
        print("{} - {}".format(file_name, data_type))
        run(file_name, data_type, check_data)

