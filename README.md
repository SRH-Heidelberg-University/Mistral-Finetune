# Mistral Finetune

Now let's begin the process of optimizing the Mistral model,

## Getting Started
To get a working instance of Mistral Finetune for development and testing purposes, follow the below instructions.

### Prerequisites
- **Anaconda**: Download and install Anaconda from the [official Anaconda website](https://www.anaconda.com/products/individual).

### Setup Instructions

#### Step 1: Clone the Repository
Use the following command to clone the Mistral repository :
```bash
git clone https://github.com/SRH-Heidelberg-University/Mistral-Finetune.git 
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
1. To save the finetune model in your hugging face profile and verify the model,

   run the below steps in terminal
    ```bash
   pip install huggingface_hub
   ```
   ```bash
   huggingface-cli login  #provide the access token of your account
   ```
   [For access token -> In your profile, under `Access Tokens` section, create a new key with `write` permission]


   [To use my account, required access details are provided in `tokens_and_model.py` file]


2. Now, lets start finetune the model,

   Run the command  `python model_train.py`.
   
   You will be notified from wandb `select option 2` and provide the `wandb api key` mentioned in `tokens_and_model.py`

4. By default, model saves in my profile.
   
   To save the model in your account, update `line 79 and 80` in `model_train.py`

    ```bash
    ft_model.push_to_hub("<your account name>/mistral-test",token="<your access token>")
   ```
   ```bash
   tokenizer.push_to_hub("<your account name>/mistral-test",token="<your access token>")
   ```

#### Step 5: Plot the loss
Collected training and validation loss for each step count (count=50) in excel file and plotted a graph with that data.
 
### Tech Stack:
- Python: Core programming language
- LangChain: Library for building language model applications
- Mistral : AI model for natural language understanding and generation
- Finetune : QLoRA,LoRA and PEFT finetuning techniques

### Acknowledgments
Special thanks to SRH Heidelberg for supporting this project.
