# Sequence Classification with Pre-trained Language model
## Requirement
- python==3.7.6
- torch==1.10.0
- transformers==4.12.5

The required packages are defined in the `requirements_cpu.txt`, or `requirements_gpu.txt`, which uses another pytorch version for gpu.
Please install the packages in your own virtual environment.
If you are using conda, use the following command:
```
pip install -r requirements_{cpu|gpu}.txt
```

## Overview
This repository is trying to reduce the task of training, or fine-tuning, of a transformer-based pre-trained 
language model, while allowing users to interchange one more encapsulated components that appear in the pipeline.
The implementation is trying hard to encapsulate different component parts as independent as possible, by means of 
having a generic interface. 

## Pipeline
The pipeline of training or fine-tuning a language model with a downstream classifier contains the following parts.
1. a Dataset class, which inherits the implementation of a [huggingface dataset](https://huggingface.co/datasets).
2. a Generator class, which the model class uses for different training schemes. For example, for regular supervised batch training, one might want a Generator to loop over the dataset batch-wise.
While for few shot learning, one might desire an episodic sampler within the Generator.
3. a Validator class, which takes the config instances as input and validate the value type, defined in the validator. This is because hydra does not support validation of values.
4. a Model class, which defines the training pipeline, given a downstream classification task.
5. an Evaluator class, which reduces the evaluation process depending on the type of output data given by the model object.

## Dataset
The huggingface Dataset class contains three methods to be customized. 
```
def _info(self):
    return datasets.DatasetInfo(...)
```
The `_info()` method defines the meta information of this dataset. This includes:
- the description of the dataset
- the class label of the dataset
- the value type for each column
- the homepage if it exists
- the citation
```
def split_generators(self):
    ## do something to get the data from a URL or a local file(s)
    ## create training/validation/testing data placeholder by assigning the path to the data
```
The `_split_generators()` method is in charge of downloading from a URL or referring to local files, 
as well as split them into various type of data, training, validation or testing.

```
def _generate_examples(self):
    ## read the data
    ## select information that you want to keep in the data 
```
The `_generate_examples()` method already has the information of the downloaded/referred data from the web/local machine.
In this method, the information selection process of reading a data is happening. For example, if your dataset has various columns
storing lots of different information, but you are only interested in some of those. You should define the information of interet here.

## Generator
The Generator class is instantiated with the config of type DictConfig defined in hydra. 

In the constructor, number of labels (`num_labels`) must be defined, as well as a `label_index_map` of type dictionary.

There are two properties in this class, which is the `training_dataset` and `testing_dataset` (also `validation_dataset` in the future).
The property loads the dataset from remote of locally, which is used in the constructor (typically) to define number of labels.

During the loading process of the dataset, a renaming of the label column will happen. This is because the pipeline expects 
the label column of all dataset to be the same. Such renaming allows the pipeline to be functioning, invariant to datasets.

The last step of customizing a Genarator to the implementation of the `__call__` method, which creates a pytorch Dataloader.
Note that, if you wish to change a sampler, the instantiated sampler object must be given as an argument (`sampler=Optional[Sampler]`).

## Validator
The validator class is responsible for validating the value in the config file. This project uses
hydra as a configuration manager. One drawback of hydra is that it does not support automatic validation of values.
This creates trouble because I want to make the error within each component to stay only in that component. Therefore a validator should 
be implemented for each type of config scheme. This component is highly customizable.

## Model
The Model class is the main blood of this repository. It constructs an `encoder` (pre-trained language model), 
a `feature_transformer` (representation learning after the encoded feature in the latent feature space, or projecting 
the features onto another space such as hyperbolic space) and a `classification_head` (whose job is to construct the classifier)

One must implement the forward method as in a pytorch fashion. The general steps will be as follows:
1. `encoder` encodes the raw text features;
2. `feature_transformer` transforms the encoded features;
3. `classification_head` makes prediction based on the transformed features.

Note that the `classification_head` by itself is also a `torch.nn.Module`, which process information per training batch/episode/a set of data.

### Methods
1. `trim_encoder_layers(encoder: PreTrainedModel, num_layers_to_keep: int) -> PreTrainedModel` 
- In some situation, it is desirable to use only part of the transformer layers in the pre-trained language model.
2. `freeze_encoder(encoder: PreTrainedModel, layers_to_freeze: Union[str, int]) -> PreTrainedModel`
- In some situation, it is desirable __NOT__ to train the encoder. For example it might help generalization in few-shot setting.
3. `preprocess(self, batch)`
- The preprocessing step does not happen within the Generator, but here. The reason for that is to have a consistent pipeline 
in both training and testing step. This is because for application, the input sentence is going directly into the Model class. 
4. `train_model(self, data_loader: DataLoader)`
- This method transfer the processing from cpu to gpu, if gpus are found. And it defines the training per epoch, as well as the evaluation and saving processing.
5. `run_per_epoch(self, data_loader: DataLoader, test: Optional[bool] = False) -> Tuple[List, List, int]`
- This method defines in details how should each training epoch look like, and outputs the information for evaluation, such as 
the predicted label and the true label, as well as the loss in this epoch.
6. `evaluate(self, y_predict: List, y_true: List, loss: int, num_epoch: Optional[int] = None) -> Tuple[float, float, str]`
- This should make use of the Evaluator object. But so far, it is not implemented yet formally.
7. `test_model(self, data_loader: DataLoader)`
- for doing inference and evaluating the performance in test time.