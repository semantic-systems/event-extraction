import csv
import os
import glob
import datasets
import pandas as pd
import numpy as np

from typing import Dict
from datasets.tasks import TextClassification
from sklearn.model_selection import train_test_split


_CITATION = """\

"""

_DESCRIPTION = """\
explainable sexism detection yeah baby
"""


class SexismLevelThreeConfig(datasets.BuilderConfig):
    """BuilderConfig for TREC IS."""

    def __init__(self, **kwargs):
        """BuilderConfig for TREC IS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SexismLevelThreeConfig, self).__init__(**kwargs)


class SexismLevelThreeDataset(datasets.GeneratorBasedBuilder):
    """TREC-IS Dataset"""

    _URL = {"train": "../../../../data/sexism/sexism_level_three_train.tsv",
            "validation": "../../../../data/sexism/sexism_level_three_validation.tsv",
            "test": "../../../../data/sexism/sexism_level_three_test.tsv"}

    train_data_label = ['none', '2_3_dehumanising_attacks_overt_sexual_objectification',
       '2_1_descriptive_attacks',
       '1_2_incitement_and_encouragement_of_harm',
       '3_1_casual_use_of_gendered_slurs_profanities_and_insults',
       '4_2_supporting_systemic_discrimination_against_women_as_a_group',
       '2_2_aggressive_and_emotive_attacks',
       '3_2_immutable_gender_differences_and_gender_stereotypes',
       '3_4_condescending_explanations_or_unwelcome_advice',
       '3_3_backhanded_gendered_compliments',
       '4_1_supporting_mistreatment_of_individual_women',
       '1_1_threats_of_harm']
    BUILDER_CONFIGS = [
        SexismLevelThreeConfig(
            name="sexism level three",
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
                    "label": datasets.features.ClassLabel(names=['none', '2_3_dehumanising_attacks_overt_sexual_objectification',
       '2_1_descriptive_attacks',
       '1_2_incitement_and_encouragement_of_harm',
       '3_1_casual_use_of_gendered_slurs_profanities_and_insults',
       '4_2_supporting_systemic_discrimination_against_women_as_a_group',
       '2_2_aggressive_and_emotive_attacks',
       '3_2_immutable_gender_differences_and_gender_stereotypes',
       '3_4_condescending_explanations_or_unwelcome_advice',
       '3_3_backhanded_gendered_compliments',
       '4_1_supporting_mistreatment_of_individual_women',
       '1_1_threats_of_harm']),
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