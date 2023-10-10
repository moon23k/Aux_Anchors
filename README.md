## Auxiliary Training for Natural Language Generation
The Transformer is the most widely used model architecture in various natural language processing tasks. However, in models designed for Natural Language Generation, training is carried out using Teacher Forcing through masking.
In practice, during the inference process for Natural Language Generation, the final inference results are generated based solely on the model's predictions. 
This discrepancy between training and inference has a detrimental effect on the model's performance.
One of the most intuitive ways to address this discrepancy is to align the training and inference processes and conduct extensive training based on a large amount of data. 
However, the former approach suffers from low training efficiency, while the latter faces challenges in data acquisition and resource-intensive training.
Therefore, in this repository, an approach is proposed that maintains efficient parallel training of a typical Transformer model while utilizing auxiliary Training Objectives to mitigate the impact of this discrepancy. 
The main objective is set as Maximum Likelihood Estimation (MLE), and the auxiliary Training Objective is First Token Prediction. 
Performance evaluation is conducted in the domains of Machine Translation, Dialogue Generation, and Text Summarization.

<br><br>

## Training Objectives

**Main Training Objectives**
> Maximum Likelihood Estimation

<br> 

**Auxiliary Training Objectives**
> First Token Prediction

<br><br> 


## Experimental Setup
| Data Setup | Model Setup | Training Setup |
|---|---|---|


<br><br>


## Results
| Aux Training Objective | Main Training Objective | Machine Translation | Dialogue Generation | Text Summarization |
|:---:|:---:|:---:|:---:|:---:|
|  0% | 100% |-|-|-|
| 10% |  90% |-|-|-|
| 30% |  70% |-|-|-|
| 50% |  50% |-|-|-|

<br><br>


## How to Use

**Clone repo in your env**
```
git clone https://github.com/moon23k/Aux_Training.git
```

**Prepare Datasets and Tokenizer via setup.py**
```
python3 setup.py -task ['all', 'translation', 'dialogue', 'summarizaiton']
```

**Main Process**
```
python3 run.py -task ['translation', 'dialogue', 'summarization']
               -mode ['train', 'test', 'inference']
               -aux_ratio ["Type percent"]
               -search ['greedy', 'beam']
```
<br>


## Reference
* [**Attention is all you need**]()

