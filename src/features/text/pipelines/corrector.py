from sklearn.pipeline import Pipeline
from src.features.text.transformers.encoding_corrector import EncodingIsolator, ExcessPunctuationRemover, HandMadeCorrector, EncodingApostropheCorrector, \
    ClassicPatternsCorrector, PronounsApostropheCorrector, DoubleEncodingTransformer, NumberEncodingCorrector, LowerTransformer, PySpellCorrector, \
    SpecificIsolator, AccentCorrector, SpaceReplacement


class CleanEncodingPipeline(Pipeline):
    """"
    Pipeline to clean encoding errors in text data.
    """

    def __init__(self):
        super().__init__(
            steps=[
                ("Isolate Special duplicates", EncodingIsolator()),
                ("Remove Excess Punctuation", ExcessPunctuationRemover()),
                ("Specific Isolation", SpecificIsolator()),
                ("Remove Double Encoding", DoubleEncodingTransformer()),
                ("Correct Bad Numbers", NumberEncodingCorrector()),
                ("Lower Text", LowerTransformer()),

                ("Correct Apostrophes", EncodingApostropheCorrector()),
                ("Correct QU Apostrophes", PronounsApostropheCorrector()),
                ("Correct Accents", AccentCorrector()),
                ("Hand Made Correction", HandMadeCorrector()),
                ("Correct Classic patterns", ClassicPatternsCorrector()),
                ("Pyspell Correction", PySpellCorrector()),
                ("Spaces Correction", SpaceReplacement())




            ]
        )
