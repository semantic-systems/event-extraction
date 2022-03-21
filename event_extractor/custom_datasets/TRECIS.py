"""Allocine Dataset: A Large-Scale French Movie Reviews Dataset."""
import csv
import os

import datasets
from datasets.tasks import TextClassification


_CITATION = """\

"""

_DESCRIPTION = """\
 TREC Incident Stream Dataset: REC-IS provides multiple Twitter datasets collected from a range of past wildfire, 
 earthquake, flood, typhoon/hurricane, bombing and shooting events (labeled pandemic data will be added later this 
 year). We have had human annotators manually label this data into 25 information types based on the information each 
 tweet contains, such as 'contains location' or is a 'search and rescue request'. Each tweet is also assigned a 
 'priority' label, that indicates how critical the information within that tweet is for a response officer to see. 
 To date, TREC-IS has manually annotated tweet streams for 48 emergency events, comprising 50,000 tweets and
  producing over 125,000 labels.
"""


class TrecisConfig(datasets.BuilderConfig):
    """BuilderConfig for TREC IS."""

    def __init__(self, **kwargs):
        """BuilderConfig for TREC IS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TrecisConfig, self).__init__(**kwargs)


class TrecisDataset(datasets.GeneratorBasedBuilder):
    """TREC-IS Dataset"""

    _URL = {"train": "../../../../data/trec_is/train.csv",
            "test": "../../../../data/trec_is/test.csv"}

    # test_data_label = ['Accident', 'shooting', 'Shooting', 'floods', 'tornadoes', 'Storm', 'Floods', 'UNASSIGNED', 'earthquake', 'Hurricane', 'Explosion', 'Wildfire', 'Flood', 'Tornado', 'Mudslide']
    train_data_label = ['wildfire', 'earthquake', 'flood', 'typhoon', 'shooting', 'bombing', 'pandemic', 'explosion', 'storm', 'fire', 'hostage', 'tornado']
    BUILDER_CONFIGS = [
        TrecisConfig(
            name="trecis",
            version=datasets.Version("1.0.0"),
            description="TREC-IS Dataset: TREC Incident Stream Dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['wildfire', 'earthquake', 'flood', 'typhoon',
                                                                 'shooting', 'bombing', 'pandemic', 'explosion',
                                                                 'storm', 'fire', 'hostage', 'tornado']),
                }
            ),
            supervised_keys=None,
            homepage="http://dcs.gla.ac.uk/~richardm/TREC_IS/2020/data.html",
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
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir["test"])}
            )
        ]

    def _generate_examples(self, filepath):
        """Generate Trec IS examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=";", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            # remove header from the generator
            _ = next(csv_reader)
            for id_, row in enumerate(csv_reader):
                _, num, _, _, _, label, _, text = row
                if num not in ["TRECIS-CTIT-H-086", "TRECIS-CTIT-H-117"]:
                    yield id_, {"text": text, "label": label}
