# TIMIT

The TIMIT database, in brief, contains audio recordings of sentences spoken
by a set of people. It also includes word and phoneme transcriptions, along with
their exact positions, as ranges, within the audio files.

As such, it is an interesting target for ML: we are given high-grade audio recordings as well as real-time phoneme and word transcriptions (or guesses at them, anyway).

The actual TIMIT database is NOT included, and is not free. Get it here:
https://catalog.ldc.upenn.edu/LDC93S1. This library merely adds
convenience, parsing, sampling, drawing, etc.

![alt text](https://github.com/colinator/timit_utils/blob/master/advert.png "Example output")



# timit_utils

The code herein can lazily load, parse, and expose the TIMIT database
of spoken audio, word and phoneme transcriptions. The layout of the TIMIT file system looks like this:

![alt text](https://github.com/colinator/timit_utils/blob/master/timitfiles.png "Your file system should look like this")

This library models the data with several classes:

* Corpus (such as '../TIMIT', contains two SubCorpuses: train and test)
* SubCorpus (such as 'train'|'test', contains several Regions)
* Region (such as 'DR1', contains several Persons)
* Person (such as 'Name:CJF0,Female')
* Sentence (such as 'SA1', contains audio, word, and phoneme transcriptions as numpy arrays)

All the above give many ways to index, iterate, parse, search, and expose the data as pandas Dataframes.

* various audio sampling, padding routines, mel filterbank frequency extractions, and a quick display system


# Installation

`pip install timit_utils`

timit_utils requires numpy, pandas, matplotlib, scipy, python_speech_features, and SoundFile.



# Example usage (i.e. in jupyter)

```code
%matplotlib inline
import timit_utils as tu
import timit_utils.audio_utils as au
import timit_utils.drawing_utils as du

corpus = tu.Corpus('../TIMIT')
sentence = corpus.train.sentences_by_phone_df('aa').sentence[0]
du.DrawVerticalPanels([du.AudioPanel(sentence.raw_audio, show_x_axis=True),
                       du.WordsPanel(sentence.words_df, sentence.raw_audio.shape[0], show_x_axis=True),
                       du.PhonesPanel(sentence.phones_df, sentence.raw_audio.shape[0])
                      ])
```

Full usage here:
https://github.com/colinator/timit_utils/blob/master/timit_utils_demonst.ipynb
