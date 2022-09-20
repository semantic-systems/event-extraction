import csv
import os
import datasets
from datasets.tasks import TextClassification


_CITATION = """\

"""

_DESCRIPTION = """\
explainable sexism detection yeah baby
"""


class SexismLevelOneConfig(datasets.BuilderConfig):
    """BuilderConfig for TREC IS."""

    def __init__(self, **kwargs):
        """BuilderConfig for TREC IS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SexismLevelOneConfig, self).__init__(**kwargs)


class SexismLevelOneDataset(datasets.GeneratorBasedBuilder):
    """TREC-IS Dataset"""

    _URL = {"train": "../../../../data/sexism/sexism_level_one_train.tsv",
            "validation": "../../../../data/sexism/sexism_level_one_validation.tsv",
            "test": "../../../../data/sexism/sexism_level_one_test.tsv"}

    train_data_label = ['not sexist', 'sexist']
    BUILDER_CONFIGS = [
        SexismLevelOneConfig(
            name="sexism level one",
            version=datasets.Version("2.0.0"),
            description=_DESCRIPTION,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['not sexist', 'sexist']),
                }
            ),
            supervised_keys=None,
            homepage="www.google.com",
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(self._URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir["train"])}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir["validation"])}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir["test"])}
            )
        ]

    def _generate_examples(self, filepath):
        """Generate Trec IS examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter="\t", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            # remove header from the generator
            _ = next(csv_reader)
            for id_, row in enumerate(csv_reader):
                try:
                    _, text, label = row
                    yield id_, {"text": text, "label": label}
                except ValueError:
                    # TODO: log and check the invalid entries
                    pass