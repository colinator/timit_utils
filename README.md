#timit_utils

The code herein can lazily load, parse, and expose the TIMIT database
of spoken audio, word and phoneme transcriptions.

Full usage here: https://github.com/colinator/timit_utils/timit_utils_demonst.ipynb


#Installation

`pip install timit_utils`

timit_utils requires numpy, pandas, matplotlib, scipy, python_speech_features, and SoundFile.

The actual TIMIT database is NOT included, and is not free. Get it here:
https://catalog.ldc.upenn.edu/LDC93S1. This library merely adds
convenience, parsing, sampling, drawing, etc.

The TIMIT database, in brief, contains audio recordings of sentences spoken
by a set of people. It also includes word and phoneme transcriptions, along with
their exact positions, as ranges, within the audio files.


#Example usage (i.e. in jupyter)

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
