from abc import ABC
import datasets
import os
import pandas as pd


class CVConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(CVConfig, self).__init__(**kwargs)


class CV(datasets.GeneratorBasedBuilder, ABC):
    BUILDER_CONFIGS = [
        CVConfig(name="Kayan Dataset", version=datasets.Version("1.0.0"), description="Kayan Dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'O',
                                'B-Degree', 'I-Degree',
                                'B-Email', 'I-Email',
                                'B-GPA', 'I-GPA',
                                'B-GPE', 'I-GPE',
                                'B-Major', 'I-Major',
                                'B-Phone', 'I-Phone',
                                'B-Skills', 'I-Skills',
                                'B-brthdate', 'I-brthdate',
                                'B-contratctype', 'I-contratctype',
                                'B-courses', 'I-courses',
                                'B-gender', 'I-gender',
                                'B-hascertificate', 'I-hascertificate',
                                'B-languages', 'I-languages',
                                'B-location', 'I-location',
                                'B-name', 'I-name',
                                'B-nationality', 'I-nationality',
                                'B-position', 'I-position',
                                'B-studiedat', 'I-studiedat',
                                'B-summary', 'I-summary',
                                'B-workat', 'I-workat'
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        base_path = "C:/Users/moham/Desktop/Stuff/Training/Projs/KayanDataset/Concated New1/"
        train_path = os.path.join(base_path, "train.csv")
        dev_path = os.path.join(base_path, "valid.csv")
        test_path = os.path.join(base_path, "test.csv")

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
        for idx, row in df.iterrows():
            id_ = row['id']
            tokens = eval(row['tokens'])
            ner_class = eval(row['ner_class'])

            yield idx, {
                "id": id_,
                "tokens": tokens,
                "ner_tags": ner_class,
            }
