import collections
import torch
from tqdm import tqdm
import datasets
import pickle
from dataset import get_dataset

# import transformers

from vocab import Vocabulary
from model import Model
from utils import Accuracy, Recall, tokenize, tokenize_instance, logger


def infer(text, model, vocab, config, class_label):
  tokens = tokenize(text)
  token_ids = vocab.map_tokens_to_ids(tokens)
  token_ids = torch.tensor(token_ids, device=config.DEVICE).unsqueeze(0)
  inputs = {'token_ids': token_ids}
  model.eval()
  output_dict = model(**inputs)
  predicted_label = output_dict['predictions'][0].item()
  predicted_label_str = class_label.int2str(predicted_label)
  logger.info(predicted_label_str)

def tokens_to_ids(instance, vocab, config):
    return {'token_ids': vocab.map_tokens_to_ids(instance['tokens'], max_length=config.MAX_LENGTH)}


def train(train_dataloader, model, optimizer, config ):
    # Training loop
    logger.info('Train')
    model.train() # THIS PART IS VERY IMPORTANT TO SET BEFORE TRAINING
    accuracy = Accuracy()
    recall = Recall()

    progress_bar = tqdm(train_dataloader)
    for batch in progress_bar:

        # Move data onto GPU.
        batch = {k: v.to(device=config.DEVICE) for k, v in batch.items()}
        
        # Zero out the gradients.
        optimizer.zero_grad()

        # Compute the model output.
        output_dict = model(**batch)

        # Backpropagate the loss and update the model weights.
        loss = output_dict['loss']
        loss.backward()
        optimizer.step()

        # Compute accuracy for monitoring
        accuracy.update(output_dict['predictions'],batch['label'])
        recall.update(output_dict['predictions'],batch['label'])

        progress_bar.set_description(f'Acc: {accuracy.get():.3f}, Recall: {recall.get():.3f}')

def validate(val_dataloader, model, config, best_val_recall ):
    logger.info('Validate')
    model.eval() # THIS PART IS VERY IMPORTANT TO SET BEFORE EVALUATION
    accuracy = Accuracy()
    recall = Recall()

    no_ans_probs={}

    progress_bar = tqdm(val_dataloader)
    for batch in progress_bar:

        # Move data onto GPU.
        batch = {k: v.to(device=config.DEVICE) for k, v in batch.items()}

        # Compute the model output.
        output_dict = model(**batch)

        # Compute accuracy for monitoring
        accuracy.update(output_dict['predictions'], batch['label'])
        recall.update(output_dict['predictions'],batch['label'])
        no_ans_probs.update({batch['id'][idx].item() : output_dict['no_answer_probability'][idx] for idx in range(len(batch['label'])) })
        progress_bar.set_description(f'Acc: {accuracy.get():.3f}, Recall: {recall.get():.3f}')

    val_recall = recall.get()
    if val_recall > best_val_recall:
        logger.info('Best so far')
        torch.save(model.state_dict(), f'checkpoints/best-model.pt')
        best_val_recall = val_recall
    return best_val_recall

def prepare_data(config):
    # train_data = datasets.load_dataset('tweet_eval', name='emoji', split='train')
    # val_data = datasets.load_dataset('tweet_eval', name='emoji', split='validation')
    # test_data = datasets.load_dataset('tweet_eval', name='emoji', split='test')
    # logger.info(train_data)
    dataset = get_dataset()
    train_data, val_data, test_data, num_classes = dataset["train"], dataset["valid"], dataset["test"], dataset["num_classes"]
    logger.info("features: " + str(train_data.info.features))

    train_data = train_data.map(tokenize_instance)
    val_data = val_data.map(tokenize_instance)
    test_data = test_data.map(tokenize_instance)


    # Count the token frequencies.
    counter = collections.Counter()
    for instance in train_data:
        counter.update(instance['tokens'])

    top_words = [x[0] for x in counter.most_common(config.VOCAB_SIZE)]

    vocab = Vocabulary(top_words, add_unk_token=True)

    train_data = train_data.map(tokens_to_ids, fn_kwargs={'vocab':vocab, 'config': config})
    val_data = val_data.map(tokens_to_ids, fn_kwargs={'vocab':vocab, 'config': config})
    test_data = test_data.map(tokens_to_ids, fn_kwargs={'vocab':vocab, 'config': config})

    train_data.set_format(type='torch', columns=['id', 'token_ids', 'label'])
    val_data.set_format(type='torch', columns=['id', 'token_ids', 'label'])
    test_data.set_format(type='torch', columns=['id', 'token_ids', 'label'])

    return train_data, val_data, test_data, vocab, num_classes


def main():

    class Config:pass
    config = Config()

    config.VOCAB_SIZE = 10000  # No magic numbers.
    config.MAX_LENGTH = 60
    config.EPOCHS = 20
    config.BATCH_SIZE = 16
    config.DROPOUT_RATE = 0.1
    config.HIDDEN_DIM = 128
    config.LEARNING_RATE = 3e-4
    config.DEVICE = 'cuda' if  torch.cuda.is_available() else 'cpu'
    logger.info(f"Running on {config.DEVICE}")


    train_data, val_data, test_data, vocab, num_classes = prepare_data(config)
    vocab_size = len(vocab)
    with open('checkpoints/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    model = Model(
        vocab_size=vocab_size,
        n_classes=num_classes,
        hidden_dim=config.HIDDEN_DIM,
        dropout_rate=config.DROPOUT_RATE
    )
    model.to(device=config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE)
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCH_SIZE)


    best_val_recall = 0.0
    for epoch in range(config.EPOCHS):
        logger.info('\nEpoch' + str(epoch))
        train(train_dataloader, model, optimizer, config)
        best_val_recall  = validate(val_dataloader, model, config, best_val_recall)


    # Load the best model checkpoint
    state_dict = torch.load(f'checkpoints/best-model.pt')
    model.load_state_dict(state_dict)
    model.to(device=config.DEVICE)
    return model, vocab, config, num_classes


if __name__ == '__main__':
    model, vocab, config, num_classes = main()
    # infer('LOL that is fuunnnyyyy.', model, vocab, config, class_label)