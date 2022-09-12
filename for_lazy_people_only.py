import abc
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List


class Table(object):
    name: str
    column: List
    benchmark_data: str

    def __init__(self, result_df: pd.DataFrame):
        self.result_df = result_df

    @abc.abstractmethod
    def write_row(self, row_name: str, task_list: List):
        raise NotImplementedError

    @abc.abstractmethod
    def write_end(self, **kwargs):
        raise NotImplementedError


class TweetEvalResultTable(Table):
    name: str = "TweetEval"
    column: List = ["", "Emoji", "Emotion", "Hate", "Irony", "Offensive", "Sentiment", "Stance", "All"]
    benchmark_data: str = "\\scalebox{0.75}{\n" \
                          "\\begin{center}\n" \
                          "\\begin{tabular}{c|c|c|c|c|c|c|c||c}\n" \
                          "\\hline\n" \
                          "&Emoji&Emotion&Hate&Irony&Offensive&Sentiment&Stance&All\\\ \n" \
                          "\\hline\\hline \n" \
                          "SVM& 29.3& 64.7&36.7&61.7&52.3&62.9&67.3&53.5\\\ \n" \
                          "FastText& 25.8& 65.2&50.6&63.1&73.4&62.9&65.4&58.1\\\ \n" \
                          "BLSTM& 24.7& 66.0&52.6&62.8&71.7&58.3&59.4&56.5\\\ \n" \
                          "Rob-Bs& 30.9\small$\pm$0.2\\thinspace(30.8)& 76.1\small$\pm$0.5\\thinspace(76.6)& 46.6\small$\pm$2.5\\thinspace(44.9)&59.7\small$\pm$5.0\\thinspace(55.2)&79.5\small$\pm$0.7\\thinspace(78.7)& 71.3\small$\pm$1.1\\thinspace(72.0)&68.0\small$\pm$0.8\\thinspace(70.9)&61.3\\\ \n" \
                          "Rob-RT& 31.4\small$\pm$0.4\\thinspace(31.6)& 78.5\small$\pm$1.2\\thinspace(79.8)& 52.3\small$\pm$0.2\\thinspace(55.5)&61.7\small$\pm$0.6\\thinspace(62.5)&80.5\small$\pm$1.4\\thinspace(81.6)& 72.6\small$\pm$0.4\\thinspace(72.9)&69.3\small$\pm$1.1\\thinspace(72.6)&65.2\\\ \n " \
                          "Rob-Tw& 29.3\small$\pm$0.4\\thinspace(29.5)& 72.0\small$\pm$0.9\\thinspace(71.7)& 46.9\small$\pm$2.9\\thinspace(45.1)&65.4\small$\pm$3.1\\thinspace(65.1)&77.1\small$\pm$1.3\\thinspace(78.6)&69.1\small$\pm$1.2\\thinspace(69.3)&66.7\small$\pm$1.0\\thinspace(67.9)&61.0\\\ \n " \
                          "XLM-R& 28.6\small$\pm$0.7\\thinspace(27.7)& 72.3\small$\pm$3.6\\thinspace(68.5)& 44.4\small$\pm$0.7\\thinspace(43.9)&57.4\small$\pm$4.7\\thinspace(54.2)&75.7\small$\pm$1.9\\thinspace(73.6)&68.6\small$\pm$1.2\\thinspace(69.6)&65.4\small$\pm$0.8\\thinspace(66.0)&57.6\\\ \n " \
                          "XLM-Tw& 30.9\small$\pm$0.5\\thinspace(30.8)& 77.0\small$\pm$1.5\\thinspace(78.3)& 50.8\small$\pm$0.6\\thinspace(51.5)&69.9\small$\pm$1.0\\thinspace(70.0)&79.9\small$\pm$0.8\\thinspace(79.3)&72.3\small$\pm$0.2\\thinspace(72.3)&67.1\small$\pm$1.4\\thinspace(68.7)&64.4\\\ \n"  \
                          "\\hline\n"

    def __init__(self, result_df: pd.DataFrame):
        super(TweetEvalResultTable, self).__init__(result_df)

    def write_row(self, row_name: str, task_list: List):
        tex = f"{row_name}&"
        scores = {task: {"avg": 0, "std": 0, "max": 0} for task in task_list if "stance" not in task}
        stance_avg_score = []
        for i, task in enumerate(task_list):
            df = self.result_df[(self.result_df['model'] == row_name) & (self.result_df['task'] == task)]
            if "stance" in task:
                stance_avg_score.append(df['metric_score'].mean())
            else:
                scores[task]["avg"] = df['metric_score'].mean()
                scores[task]["std"] = df['metric_score'].std()
                scores[task]["max"] = df['metric_score'].max()
        scores["stance"] = {"avg": 0, "std": 0, "max": 0}
        scores["stance"]["avg"] = np.mean(stance_avg_score)
        scores["stance"]["std"] = np.std(stance_avg_score)
        scores["stance"]["max"] = np.max(stance_avg_score)
        scores["all"] = np.mean([task_dict["avg"] for task_dict in scores.values()])
        scores = {k.capitalize(): v for k, v in scores.items()}
        latex_column = self.column[1:-1]
        for i, task in enumerate(latex_column):
            tex += f"{round(100*scores[task]['avg'], 1)}\small$\pm${round(100*scores[task]['std'], 1)}\\thinspace({round(100*scores[task]['max'], 1)})&\n"
            if i == len(latex_column)-1:
                tex += str(round(100*scores["All"], 1))
                tex += "\\\ \n"
        return tex

    def write_end(self) -> str:
        string = "\\hline\\hline\n" \
                 "\\textbf{Metric}&M-F1&M-F1&M-F1&F$^{(i)}$&M-F1&M-Rec&AVG(F$^{(a)}$, F$^{(f)}$)&TE\n" \
                 "\\end{tabular}\n" \
                 "\\end{center}}"
        return string


class CrisisResultTable(Table):
    name: str = "Crisis"
    column: List = ["", "Event Type"]
    benchmark_data: str = "\\scalebox{0.75}{\n" \
                          "\\begin{center}\n" \
                          "\\begin{tabular}{c|c}\n" \
                          "\\hline\n" \
                          "&Event Type\\\ \n" \
                          "\\hline\\hline \n"

    def __init__(self, result_df: pd.DataFrame):
        super(CrisisResultTable, self).__init__(result_df)

    def write_row(self, row_name: str, task_list: List):
        tex = f"{row_name}&"
        scores = {task: {"avg": 0, "std": 0, "max": 0} for task in task_list}
        for i, task in enumerate(task_list):
            df = self.result_df[(self.result_df['model'] == row_name) & (self.result_df['task'] == task)]
            scores[task]["avg"] = df['metric_score'].mean()
            scores[task]["std"] = df['metric_score'].std()
            scores[task]["max"] = df['metric_score'].max()
        for i, task in enumerate(self.column[1:]):
            tex += f"{round(100*scores[task]['avg'], 1)}\small$\pm${round(100*scores[task]['std'], 1)}\\thinspace({round(100*scores[task]['max'], 1)})&\n"
            tex += "\\\ \n"
        return tex

    def write_end(self) -> str:
        string = "\\hline\\hline\n" \
                 "\\textbf{Metric}&M-F1\n" \
                 "\\end{tabular}\n" \
                 "\\end{center}}"
        return string


class Result(object):
    def __init__(self, path: str, root: str):
        self.root = root
        self.path = path
        self.result = self.read_json(path)
        self.tasks = []
        self.seeds = [0, 1, 2]

    @staticmethod
    def read_json(path: str):
        with open(path, "r") as f:
            d = json.load(f)
        return d

    def get_seed(self) -> int:
        for seed in self.seeds:
            if f"seed_{seed}" in self.path:
                return seed

    def get_task(self) -> str:
        for task in self.tasks:
            if task in self.path:
                return task

    def get_model(self) -> str:
        return re.search(f"{self.root}(.*)/", self.path).group(1).split("/")[0]

    @staticmethod
    def get_metric_name(task):
        raise NotImplementedError

    def get_metric(self, metric_name) -> float:
        if len(metric_name) == 1:
            metric = self.result[0].get(metric_name[0])
        elif len(metric_name) == 2:
            metric = self.result[0].get(metric_name[0]).get(metric_name[1])
        else:
            raise ValueError
        return metric


class CrisisResult(Result):
    def __init__(self, path: str, root: str):
        super(CrisisResult, self).__init__(path, root)
        self.tasks = ["Event Type"]
        self.seeds = [0, 1, 2]
        self.task = "Event Type"
        self.seed = self.get_seed()
        self.model = self.get_model()
        self.metric_name = self.get_metric_name(self.task)
        self.metric = self.get_metric(self.metric_name)

    @staticmethod
    def get_metric_name(task):
        metric_dict = {"Event Type": ["f1_macro"]}
        return metric_dict[task]


class TweetEvalResult(Result):
    def __init__(self, path: str, root: str):
        super(TweetEvalResult, self).__init__(path, root)
        self.tasks = ["stance_atheism", "stance_feminist", "stance_climate", "stance_abortion", "stance_hillary",
                      "offensive", "sentiment", "hate", "irony", "emotion", "emoji"]
        self.seeds = [0, 1, 2]
        self.seed = self.get_seed()
        self.task = self.get_task()
        self.model = self.get_model()
        self.metric_name = self.get_metric_name(self.task)
        self.metric = self.get_metric(self.metric_name)

    def get_task(self) -> str:
        for task in self.tasks:
            if task in self.path:
                return task

    @staticmethod
    def get_metric_name(task):
        metric_dict = {"stance": ["other"],
                       "sentiment": ["recall_macro"],
                       "offensive": ["f1_macro"],
                       "irony": ["f1_per_class", "irony"],
                       "hate": ["f1_macro"],
                       "emotion": ["f1_macro"],
                       "emoji": ["f1_macro"]}

        task = "stance" if "stance" in task else task
        return metric_dict[task]


class LatexTableWriter(object):
    def __init__(self, output_path: str, table_class: type(Table), result_class: type(Result)):
        self.output_path = output_path
        self.result_class = result_class
        test_result_path = self.fetch_test_results_from_dir(output_path)
        self.result_instances = self.retrieve_result_instance(test_result_path)
        self.row_list = self.get_model_list(test_result_path)
        self.task_list = self.result_instances[0].tasks
        self.result_df = self.get_result_df(self.result_instances)
        self.table = table_class(self.result_df)
        self.write_to_csv(self.result_df, str(Path(self.output_path, "results.csv").absolute()))

    def get_model_list(self, path_list: list) -> list:
        return list(set([re.search(f"{self.output_path}(.*)/", path).group(1).split("/")[0] for path in path_list]))

    @staticmethod
    def write_to_csv(df: pd.DataFrame, path: str):
        df.to_csv(path)

    def fetch_test_results_from_dir(self, root: str):
        return [fname for fname in self.walk_through_files(root, "test_result.json")]

    def retrieve_result_instance(self, result_path: List[str]) -> List[Result]:
        return [self.result_class(file, root=self.output_path) for file in result_path]

    @staticmethod
    def walk_through_files(path, file_extension='.csv'):
        for (dirpath, dirnames, filenames) in os.walk(path):
            for filename in filenames:
                if filename.endswith(file_extension):
                    yield os.path.join(dirpath, filename)

    @staticmethod
    def get_result_df(result_instances: List[Result]) -> pd.DataFrame:
        metric_name = []
        for metric in [result.metric_name for result in result_instances]:
            if len(metric) == 1:
                if metric[0] == "other":
                    metric_name.append("f1_macro (averaging of favor and against class)")
                else:
                    metric_name.append(metric[0])
            elif len(metric) == 2:
                metric_name.append(f"{metric[0]}: {metric[1]}")
            else:
                raise NotImplementedError
        data = {'seed': [result.seed for result in result_instances],
                'task': [result.task for result in result_instances],
                'model': [result.model for result in result_instances],
                'metric_name': metric_name,
                'metric_score': [result.metric for result in result_instances]
                }
        return pd.DataFrame(data)

    def write_to_tex(self):
        with open(str(Path(self.output_path, f"{self.table.name}_latex_table.tex").absolute()), "w") as f:
            f.write(self.table.benchmark_data)
            for model in self.row_list:
                f.write(self.table.write_row(row_name=model, task_list=self.task_list))
            f.write(self.table.write_end())


class ConfigWriter(object):
    @staticmethod
    def write_from_dict(dict_to_dump: Dict, path_to_dump: str):
        with open(path_to_dump, 'w') as f:
            yaml.dump(dict_to_dump, f, default_flow_style=False)

    @staticmethod
    def read_yaml(yaml_to_read: str) -> Dict:
        with open(yaml_to_read, 'r') as f:
            data = yaml.safe_load(f)
        return data

    @staticmethod
    def change_field_of_all(dir: str):
        files = [os.path.join(path, name) for path, subdirs, files in os.walk(dir) for name in files if name.endswith(".yaml")]
        updated_dicts: List[Dict] = []
        for file in files:
            config = ConfigWriter.read_yaml(file)
            # config["seed"] = [0, 1, 2]
            config["model"]["output_path"] = "./outputs/tweeteval/experiments/contrastive_loss_ratio/07/"
            config["model"]["contrastive"]["contrastive_loss_ratio"] = 0.7
            updated_dicts.append(config)
            ConfigWriter.write_from_dict(config, file)


if __name__ == "__main__":
    ConfigWriter.change_field_of_all("./event_extractor/configs/tweeteval/experiments/contrastive_loss_ratio/07/")
    # writer = LatexTableWriter("./tables/tweeteval/", TweetEvalResultTable, TweetEvalResult)
    # writer.write_to_tex()
    # writer = LatexTableWriter("./tables/crisis/", CrisisResultTable, CrisisResult)
    # writer.write_to_tex()


