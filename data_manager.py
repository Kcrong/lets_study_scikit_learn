# coding: utf-8

import os.path as path
import pandas as pd


class ConversationReader:
    """
    Parsing from the Corpus data.

    Running by lazy. Fit-in at Tracy's Corpus format

    Parameters
    ----------
    file_path: str
        path information of corpus file.

    delimiter: str, optional (default=';').
        string for identify between sentence and tag

    Attributes
    ----------
    input_file_path : str
        Path to the corpus file.

    delimiter : str
        Symbol to use to separate values in records
    """

    def __init__(self, file_path, delimiter=';'):

        self.input_file_path = file_path
        self.name = path.basename(file_path)

        self.delimiter = delimiter

    def __repr__(self):
        return "<CorpusFormatter %s>" % self.name

    def _read(self):
        """
        Parsing Corpus files to sentences and tags
        """
        try:
            data = pd.read_csv(self.input_file_path, delimiter=self.delimiter)
        except FileNotFoundError:
            raise FileNotFoundError(
                "File '%s' not found. You should download it manually or use"
                " `data/download.py`. See `README.md` for more information." %
                self.input_file_path)

        agg = (data[['message_id', 'chat_id', 'data', 'category']]
               .groupby('chat_id')['data', 'category']
               .agg({'data': lambda x: "\n".join(x), 'category': max})
               .reset_index())

        labels = list(agg['category'].values)
        sentences = list(agg['data'].values)

        self._labels = labels
        self._sentences = sentences
        self._data = agg

    def get_sentences(self):
        if not hasattr(self, '_data'):
            self._read()
        return self._sentences

    def get_labels(self):
        if not hasattr(self, '_data'):
            self._read()
        return self._labels

    @property
    def sentences(self):
        return self.get_sentences()

    @property
    def labels(self):
        return self.get_labels()


def load_dataset(file_path, delimiter=';'):
    reader = ConversationReader(file_path, delimiter)
    sentences = reader.get_sentences()
    labels = reader.get_labels()

    return sentences, labels
