from typing import List

from nltk import sent_tokenize


def split_query_answer(
    query_answer: str,
    speaker_list: List[str]
) -> List[str]:
    """Split query answer from QMSum

    Only using `nltk.sent_tokenize` will split sentences wrongly.
    By using `speaker_list` and latin abbreviation(i.e. and e.g.),
    You can split sentences more correctly.

    Args:
        query_answer (str): Each answer of query in QMSum
        speaker_list (List[str]): Name of speakers

    Example:
        sentence = '''Mr. John. the linguist, is skilled in multiple
            languages (i.e.), he's fluent in other languages. Mr. John
            gave Susan. the CEO flowers (e.g.), rose, tulips, iris.'''

        * using only `sent_tokenize`

        >>> from nltk import sent_tokenize
        >>> sent_tokenize(sentence)

        Mr. John.

        the linguist, is skilled in multiple languages (i.e.

        ), he's fluent in other languages.

        Mr. John gave Susan.

        the CEO flowers (e.g.

        ), rose, tulips, iris.

        * using `split_query_answer`

        >>> from sentence_split import split_query_answer
        >>> split_query_answer(sentence)

        Mr. John. the linguist, is skilled in multiple languages (i.e. ), he's fluent in other languages.

        Mr. John gave Susan. the CEO flowers (e.g. ), rose, tulips, iris.

    Returns:
        List[str]: splited query answer
    """
    sentence_list = sent_tokenize(query_answer)

    correct_sentence_list = [sentence_list[0]]

    for sentence in sentence_list[1:]:
        prev = correct_sentence_list[-1].split()[-1]
        next_ = sentence.split()[0]

        if _check_wrong_split(prev, next_, speaker_list):
            correct_sentence_list[-1] += f' {sentence}'
        else:
            correct_sentence_list.append(sentence)

    return correct_sentence_list


def _check_wrong_split(
    first: str,
    second: str,
    speaker_list: List[str]
) -> bool:
    """Check if the sentence is wrongly splited
    """
    if 'e.g.' in first or 'i.e.' in first:
        return True

    for speaker in speaker_list:
        if f'{first} {second}' in speaker:
            return True

    return False
