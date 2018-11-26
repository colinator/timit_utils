import typing
from typing import *

import os
from collections import namedtuple, Counter

import pandas as pd
import numpy as np
import soundfile as sf

from timit_utils.basic_utils import *


'''
The code herein can lazily load, parse, and expose the TIMIT database
of spoken audio, word and phoneme transcriptions.

The actual TIMIT database is NOT included, and is not free. Get it
here: https://catalog.ldc.upenn.edu/LDC93S1. This library merely adds
convenience, parsing, sampling, drawing, etc.

Programmed with python static typing! Just cuz.

Where possible, exposes pandas Dataframes containing various views of data.

Skip to the end to find out the one initialiation step.
'''


class NameRangeSequence:
    '''Contains two np arrays: names (words or phonemes),
       and corresponding (start, stop) integer ranges.
       I.e. phoneme 'fu' starts at 8345ms in and stops at 8398ms in.
    '''
    def __init__(self, names, ranges):
        self.names = names      # np.array
        self.ranges = ranges    # np.array(n, 2)

    def create_dataframe(self): # -> pd.DataFrame:
        return pd.DataFrame(self.ranges.T, columns=['start', 'end'], index=self.names)

    @staticmethod
    def read_words_from(filepath: str) -> 'NameRangeSequence':
        '''Loads TIMIT transcriptions of spoken words and
        (or phonemes) and time ranges.'''

        with open(filepath, 'r') as f:
            words = [parse_word_triple(l) for l in f.readlines()]
            realwords = [word for word in words if word[0] != 'h#']
            names = [word.word for word in realwords]
            starts = [word.start for word in realwords]
            ends = [word.stop for word in realwords]
            return NameRangeSequence(np.array(names), np.array([starts, ends]))


class Sentence:
    '''Represents one sentence spoken by one person in the TIMIT data set:
       lazily reads raw audio, sample rate, word transcriptions,
       and phoneme transcriptions.

       Exposes the raw audio as numpy array of floats, sample_rate as
       int, and the transcriptions as either NameRangeSequences or
       pandas dataframe.

       Also exposes Counter objects for words and phonemes
       in transcription.'''

    def __init__(self, directory_path: str, name: str) -> None:
        self.directory_path = directory_path
        self.name = name
        self.full_path = os.path.join(directory_path, name)

    def __str__(self) -> str:
        return '{}'.format(self.name,)

    @lazy_property
    def _raw_data_tuple(self) -> Tuple[np.ndarray, int]:
        return sf.read(self.full_path + '.WAV')

    @lazy_property
    def raw_audio(self) -> np.ndarray:
        return self._raw_data_tuple[0]

    @lazy_property
    def sample_rate(self) -> int:
        return self._raw_data_tuple[1]

    @lazy_property
    def words_fullfreq(self) -> NameRangeSequence:
        return NameRangeSequence.read_words_from(self.full_path + '.WRD')

    @lazy_property
    def phones_fullfreq(self) -> NameRangeSequence:
        return NameRangeSequence.read_words_from(self.full_path + '.PHN')

    @lazy_property
    def words_df(self) -> pd.DataFrame:
        return self.words_fullfreq.create_dataframe()

    @lazy_property
    def phones_df(self) -> pd.DataFrame:
        return self.phones_fullfreq.create_dataframe()

    @lazy_property
    def word_counts(self): # -> typing.Counter[str]:
        return Counter(self.words_fullfreq.names)

    @lazy_property
    def phone_counts(self): # -> typing.Counter[str]:
        return Counter(self.phones_fullfreq.names)

    def print_all(self, indent_level:int=0) -> None:
        print('{}Sentence {} {}'.format(indent(indent_level), self.name, self.directory_path))

    @staticmethod
    def get_sentences_in_directory(directory: str) -> List['Sentence']:
        return [Sentence(directory, d) for d in get_sentence_names_in(directory)]


class Person:
    '''Represents a person object in TIMIT world, and can lazily load of their
       audio recordings and transcriptions.

       Offers various ways to index into and find sentences.'''

    def __init__(self, directory_path: str, person_directory_name: str, region_name: str) -> None:
        self.directory_path = directory_path
        self.person_directory_name = person_directory_name
        self.region_name = region_name
        self.full_path = os.path.join(directory_path, person_directory_name)
        self.name = person_directory_name[1:]
        self.gender = person_directory_name[0]

    def __str__(self) -> str:
        return '{} ({})'.format(self.name, self.gender)

    @lazy_property
    def sentences(self) -> Dict[str, Sentence]:
        return { s.name: s for s in Sentence.get_sentences_in_directory(self.full_path) }

    @lazy_property
    def sentence_names(self) -> List[str]:
        return sorted([self.sentences[s].name for s in self.sentences])

    def num_sentences(self) -> int:
        return len(self.sentences)

    def sentence_by_index(self, index: int) -> Sentence:
        return self.sentences[self.sentence_names[index]]

    def sentence_by_name(self, name: str) -> Sentence:
        return self.sentences[name]

    @lazy_property
    def words_to_sentences_and_counts(self) -> StringsToSentencesAndCounts:
        return sentence_words_to_sentences_and_counts(self.sentences.values())

    @lazy_property
    def word_counts(self) -> StringsToCounts:
        return string_counts_from_strings_to_sentences_and_counts(self.words_to_sentences_and_counts)

    @lazy_property
    def phones_to_sentences_and_counts(self) -> StringsToSentencesAndCounts:
        return sentence_phones_to_sentences_and_counts(self.sentences.values())

    @lazy_property
    def phone_counts(self) -> StringsToCounts:
        return string_counts_from_strings_to_sentences_and_counts(self.phones_to_sentences_and_counts)

    def sentences_by_word(self, word: str) -> FrozenSet[Sentence]:
        return self.words_to_sentences_and_counts[word]

    def sentences_by_phone(self, phone: str) -> FrozenSet[Sentence]:
        return self.phones_to_sentences_and_counts[phone]

    def print_all(self, indent_level:int=0) -> None:
        print('{}Person {} {} {}'.format(indent(indent_level), self.name, self.gender, self.full_path))
        for s in sorted(self.sentences.keys()):
            self.sentences[s].print_all(indent_level+1)


class Region:
    '''Represents a region in the TIMIT world. Can lazily load all the people therein.

       Offers various ways to index into and find people, and get aggregation dataframes.'''

    def __init__(self, directory_path: str, name: str) -> None:
        self.directory_path = directory_path
        self.name = name
        self.full_path = os.path.join(directory_path, name)

    def __str__(self) -> str:
        return '{}'.format(self.name,)

    @staticmethod
    def _get_people_in_region(region_path: str, region_name: str) -> List[Person]:
        return [Person(region_path, d, region_name) for d in get_subdirectories_in(region_path)]

    @lazy_property
    def people(self) -> Dict[str, Person]:
        return { p.name: p for p in Region._get_people_in_region(self.full_path, self.name) }

    @lazy_property
    def people_names(self) -> List[str]:
        return sorted([self.people[p].name for p in self.people])

    def num_people(self) -> int:
        return len(self.people)

    def person_by_index(self, index: int) -> Person:
        return self.people[self.people_names[index]]

    def person_by_name(self, name: str) -> Person:
        return self.people[name]

    def print_all(self, indent_level:int=0) -> None:
        print('{}Region {} {} {}'.format(indent(indent_level), self.name, self.directory_path, self.full_path))
        for p in sorted(self.people.keys()):
            self.people[p].print_all(indent_level+1)

    @lazy_property
    def words_to_usages(self) -> StringsToPersonSentencesAndCounts:
        return people_words_to_usages(self.people.values())

    @lazy_property
    def phones_to_usages(self)-> StringsToPersonSentencesAndCounts:
        return people_phones_to_usages(self.people.values())

    @lazy_property
    def words_to_usages_df(self) -> pd.DataFrame:
        return strings_to_usages_to_df(self.words_to_usages, lambda person: person.word_counts)

    @lazy_property
    def phones_to_usages_df(self) -> pd.DataFrame:
        return strings_to_usages_to_df(self.phones_to_usages, lambda person: person.phone_counts)

    def sentences_by_word(self, word: str) -> FrozenSet[PersonSentenceCount]:
        return self.words_to_usages[word]

    def sentences_by_phone(self, phone: str) -> FrozenSet[PersonSentenceCount]:
        return self.phones_to_usages[phone]

    def sentences_by_word_df(self, word: str) -> pd.DataFrame:
        return sentence_person_counts_df(self.sentences_by_word(word))

    def sentences_by_phone_df(self, phone: str) -> pd.DataFrame:
        return sentence_person_counts_df(self.sentences_by_phone(phone))



class SubCorpus:
    '''A SubCorpus contains a dictionary of name to Region object

       It also has a few ways to extract regions, people, sentences,
       and various aggregation dataframes.'''

    @staticmethod
    def _get_regions_in_subcorpus(subcorpus_path: str) -> List[Region]:
        return [Region(subcorpus_path, subdirname) for subdirname in get_subdirectories_in(subcorpus_path)]

    def __init__(self, path: str, name: str) -> None:
        self.path = path
        self.name = name
        self.regions = { r.name: r for r in SubCorpus._get_regions_in_subcorpus(self.path) }
        self.region_names = sorted([self.regions[r].name for r in self.regions])

    def __str__(self) -> str:
        return '{}'.format(self.name,)

    def region_by_index(self, index: int) -> Region:
        return self.regions[self.region_names[index]]

    def region_by_name(self, name: str) -> Region:
        return self.regions[name]

    @lazy_property
    def people(self) -> Dict[str, Person]:
        return { p.name: p for r in self.regions.values() for p in r.people.values() }

    @lazy_property
    def people_names(self) -> List[str]:
        return sorted(list(set(self.people.keys())))

    def person_by_index(self, index: int) -> Person:
        return self.people[self.people_names[index]]

    def person_by_name(self, name: str) -> Person:
        return self.people[name]

    @lazy_property
    def words_to_usages(self) -> StringsToPersonSentencesAndCounts:
        return people_words_to_usages(self.people.values())

    @lazy_property
    def phones_to_usages(self) -> StringsToPersonSentencesAndCounts:
        return people_phones_to_usages(self.people.values())

    @lazy_property
    def words_to_usages_df(self) -> pd.DataFrame:
        return strings_to_usages_to_df(self.words_to_usages, lambda p: p.word_counts)

    @lazy_property
    def phones_to_usages_df(self) -> pd.DataFrame:
        return strings_to_usages_to_df(self.phones_to_usages, lambda p: p.phone_counts)

    def sentences_by_word(self, word: str) -> FrozenSet[PersonSentenceCount]:
        return self.words_to_usages[word]

    def sentences_by_phone(self, phone: str) -> FrozenSet[PersonSentenceCount]:
        return self.phones_to_usages[phone]

    def sentences_by_word_df(self, word: str) -> pd.DataFrame:
        return sentence_person_counts_df(self.sentences_by_word(word))

    def sentences_by_phone_df(self, phone: str) -> pd.DataFrame:
        return sentence_person_counts_df(self.sentences_by_phone(phone))

    def print_all(self, indent_level:int=0) -> None:
        print('{}SubCorpus {} {}'.format(indent(indent_level), self.name, self.path))
        for r in sorted(self.regions.keys()):
            self.regions[r].print_all(indent_level+1)



class Corpus:
    '''It all starts here!

        corpus = tu.Corpus('/your/path/to/TIMIT/')

        Does nothing but exposes two SubCorpuses: train and test.
        Everything should be 'efficient' as far as lazy-loading can take us.
    '''
    def __init__(self, path: str) -> None:
        self.path = path
        self.train = SubCorpus(os.path.join(self.path, 'TRAIN'), 'TRAIN')
        self.test = SubCorpus(os.path.join(self.path, 'TEST'), 'TEST')

    def __str__(self) -> str:
        return 'Corpus {}'.format(self.path,)

    def print_all(self, indent_level: int=0) -> None:
        print('{}Corpus {}'.format(indent(indent_level), self.path))
        self.train.print_all(indent_level+1)
        self.test.print_all(indent_level+1)
