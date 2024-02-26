# Mistral Finetune

Let's get started on how to finetune the mistral model,

## Getting Started
To get a working instance of Mistral Finetune for development and testing purposes, follow the below instructions.

### Prerequisites
- **Anaconda**: Download and install Anaconda from the [official Anaconda website](https://www.anaconda.com/products/individual).

### Setup Instructions

#### Step 1: Clone the Repository
Use the following command to clone the Mistral repository :
```bash
git clone 
```

#### Step 2: Create a Conda Environment
Navigate to the project directory and use Python 3.8 to build a new Conda environment called `nenv`
```bash
conda create -n nenv python=3.8 -y
```
Activate the environment:
```bash
conda activate nenv
```

#### Step 3: Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Also, install a few more packages like
```bash
pip install datasets
```
```bash
pip install peft
```
```bash
pip install wandb
```
#### Step 4: Start Finetuning the model and train it
Run the `python model_train.py` and you will be notified from wandb `select option 2` and provide the wandb api key mentioned in `tokens_and_model.py`

#### Step 5: Plot the loss
Collected training and validation loss for each step count (count=50) in excel file and plotted a graph with that data.
 
### Tech Stack:
- Python: Core programming language
- LangChain: Library for building language model applications
- Mistral : AI model for natural language understanding and generation

### Acknowledgments
Special thanks to SRH Heidelberg for supporting this project.
