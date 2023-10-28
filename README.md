## Auxiliary Training for Natural Language Generation

&nbsp; Transformer-based deep learning models for natural language generation tasks employ masking and teacher forcing during the training phase. 
However, during actual inference, they only utilize model predictions without teacher forcing. 
This gap between training and inference causes negative impact on model performances.
There are two approaches to address this discrepancy.
The first is to train large-scale models on extensive datasets. The second is to align the training and inference processes for generative learning. 
However, the first approach demands substantial computational resources, and the second approach suffers from the inefficiency of the training process.
To tackle this issue, in this repository, we propose a method that maintains the main training objective as MLE while additionally incorporating an auxiliary training objective called "First Token Prediction." 
We then evaluate the performance of this approach across three natural language generation tasks.

<br><br>


## Experimental Setup
| Data Setup | Model Setup | Training Setup |
|---|---|---|
||||

<br><br>


## Results
| Aux Training Ratio | Machine Translation | Dialogue Generation | Text Summarization |
|:---:|:---|:---|:---|
| 0.0 | **`BLEU:`** 16.18<br>**`First Token Prediction:`** 61.72 | **`ROUGE:`** 1.63<br>**`First Token Prediction:`** 27.34 | **`ROUGE:`** 00.00<br>**`First Token Prediction:`** |
| 0.1 | **`BLEU:`** 15.50<br>**`First Token Prediction:`** 60.94 | **`ROUGE:`** 1.43<br>**`First Token Prediction:`** 19.53 | **`ROUGE:`** 00.00<br>**`First Token Prediction:`** |
| 0.3 | **`BLEU:`**  6.84<br>**`First Token Prediction:`** 52.34 | **`ROUGE:`** 1.53<br>**`First Token Prediction:`** 19.34 | **`ROUGE:`** 00.00<br>**`First Token Prediction:`** |
| 0.5 | **`BLEU:`**  2.13<br>**`First Token Prediction:`** 59.38 | **`ROUGE:`** 1.23<br>**`First Token Prediction:`** 18.75 | **`ROUGE:`** 00.00<br>**`First Token Prediction:`** |

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
<br><br> 


## Reference
* [**Attention is all you need**]()

