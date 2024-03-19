from sklearn.pipeline import Pipeline
from src.features.text.transformers.cleaners import HtmlCleaner, LxmlCleaner, UrlCleaner, FileNameCleaner, BadHTMLCleaner, SpaceBeforeAdder, SpaceAroundAdder, ShortTextCleaner, ExtraSpacesCleaner


class CleanTextPipeline(Pipeline):
    """
    A pipeline to clean text data.
    """

    def __init__(self):
        super().__init__(
            steps=[
                ("Clean HTML", HtmlCleaner()),
                ("Clean LXML", LxmlCleaner()),
                ("Clean URLs", UrlCleaner()),
                ("Clean File names", FileNameCleaner()),
                ("Clean Bad HTML", BadHTMLCleaner()),
                ("Add Spaces Before specific patterns", SpaceBeforeAdder()),
                ("Clean Extra Spaces", ExtraSpacesCleaner()),
                ("Remove short text", ShortTextCleaner()),

            ]
        )
