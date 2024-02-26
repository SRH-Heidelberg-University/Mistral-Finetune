from datasets import load_dataset

#dataset
train_dataset = load_dataset('gem/viggo', split='train')
eval_dataset = load_dataset('gem/viggo', split='validation')
test_dataset = load_dataset('gem/viggo', split='test')

# print("Target Sentence: " + test_dataset[1]['target'])
# print("Meaning Representation: " + test_dataset[1]['meaning_representation'] + "\n")
