import typing
from typing import *

import os
from collections import namedtuple, Counter

import pandas as pd
import numpy as np
import soundfile as sf


# pure utilities

T = typing.TypeVar('T')
def flatten(arr: List[List[T]]) -> List[T]:
    return [l for sublist in arr for l in sublist]

def lazy_property(fn: Callable[[Any],Any]) -> Any:
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

def indent(level: int) -> str:
    return '  ' * level


# file and directory listing

def get_subdirectories_in(path: str) -> List[str]:
    return sorted([name for name in os.listdir(path) if not name.startswith('.')])

def get_sentence_names_in(path: str) -> List[str]:
    files = sorted([name for name in os.listdir(path) if not name.startswith('.') and name[-4:] == '.WAV'])
    return [f[:-4] for f in files]



# types we'll need

TupleWordTriple: Tuple[str, int, int] = namedtuple('TupleWordTriple', ['word', 'start', 'stop'])
TupleSentenceAndCount: Tuple['Sentence', int] = namedtuple('TupleSentenceAndCount', ['sentence', 'count'])
TuplePersonSentenceCount: Tuple['Person', 'Sentence', int] = namedtuple('TuplePersonSentenceCount', ['person', 'sentence', 'count'])

class WordTriple(TupleWordTriple):
    def __repr__(self) -> str:
        return '{} {} {}'.format(self.word, self.start, self.stop)

class SentenceAndCount(TupleSentenceAndCount):
    def __repr__(self) -> str:
        return '{}|{}'.format(self.sentence.name, self.count)

class PersonSentenceCount(TuplePersonSentenceCount):
    def __repr__(self) -> str:
        return '{}|{}|{}'.format(self.person.name, self.sentence.name, self.count)

StringsToCounts = Dict[str, int]
StringCount = Tuple[str, int]
StringCountList = List[StringCount]

StringsToSentencesAndCounts = Dict[str, FrozenSet[SentenceAndCount]]
StringsToPersonSentencesAndCounts = Dict[str, FrozenSet[PersonSentenceCount]]

SentenceStringCountsExtractor = Callable[['Sentence'], StringCountList]
PersonStringCountsExtractor = Callable[['Person'], StringsToSentencesAndCounts]


# parsing from file

def parse_word_triple(line: str) -> WordTriple:
    v = line.split(' ')
    return WordTriple(word=v[2][:-1], start=int(v[0]), stop=int(v[1]))


# word and phoneme counting by sentences

def sentences_to_sentences_and_string_counts(sentences: Sequence['Sentence'],
                                             sentence_counts_extractor: SentenceStringCountsExtractor) -> StringsToSentencesAndCounts:
    s2s: Dict[str, List[SentenceAndCount]] = {}
    for sentence in sentences:
        string_counts: StringCountList = sentence_counts_extractor(sentence)
        for string, count in string_counts:
            if string not in s2s: s2s[string] = []
            s2s[string].append(SentenceAndCount(sentence=sentence, count=count))
    return { string: frozenset(s2s[string]) for string in s2s }

def sentence_words_to_sentences_and_counts(sentences: Sequence['Sentence']) -> StringsToSentencesAndCounts:
    return sentences_to_sentences_and_string_counts(sentences, lambda s: s.word_counts.items())

def sentence_phones_to_sentences_and_counts(sentences: Sequence['Sentence']) -> StringsToSentencesAndCounts:
    return sentences_to_sentences_and_string_counts(sentences, lambda s: s.phone_counts.items())


def string_counts_from_strings_to_sentences_and_counts(s2s: StringsToSentencesAndCounts) -> StringsToCounts:
    return { s: sum([sc.count for sc in s2s[s]]) for s in s2s }


# word and phoneme counting by people

def people_to_strings_to_usages(people: Sequence['Person'], person_string_counts_extractor: PersonStringCountsExtractor) -> StringsToPersonSentencesAndCounts:
    s2u: Dict[str, List[PersonSentenceCount]] = {}
    for p in people:
        sp: StringsToSentencesAndCounts = person_string_counts_extractor(p)
        for string, sentences_and_counts in sp:
            if string not in s2u: s2u[string] = []
            for s, c in sentences_and_counts:
                s2u[string].append(PersonSentenceCount(person=p, sentence=s, count=c))
    return { string: frozenset(s2u[string]) for string in s2u }

def people_words_to_usages(people: Sequence['Person'])-> StringsToPersonSentencesAndCounts:
    return people_to_strings_to_usages(people, lambda p: p.words_to_sentences_and_counts.items())

def people_phones_to_usages(people: Sequence['Person'])-> StringsToPersonSentencesAndCounts:
    return people_to_strings_to_usages(people, lambda p: p.phones_to_sentences_and_counts.items())


# pandas DataFrame creation helper functions

def strings_to_usages_to_df(w2u: StringsToPersonSentencesAndCounts,
                            person_string_counts_extractor: Callable[['Person'], StringsToCounts]) -> pd.DataFrame:
    all_people = sorted(list(set([ psc.person for pscs in w2u.values() for psc in pscs ])), key=lambda p: p.name)
    all_strings = sorted(w2u.keys())
    data = [ [ person_string_counts_extractor(person).get(string, 0) for string in all_strings ] for person in all_people ]
    return pd.DataFrame(np.array(data), index=all_people, columns=all_strings)

def sentence_person_counts_df(sentence_person_counts: FrozenSet[PersonSentenceCount]) -> pd.DataFrame:
    ordered_list = sorted(list(sentence_person_counts), key=lambda psc: (psc.person.name, psc.sentence.name))
    people = [psc.person for psc in ordered_list]
    sentences = [psc.sentence for psc in ordered_list]
    counts = [psc.count for psc in ordered_list]
    return pd.DataFrame(np.array([sentences, counts]).T, index=people, columns=['sentence', 'count'])
