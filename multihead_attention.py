import collections
import torch
from tqdm import tqdm
import pickle
from dataset import prepare_qa_dataset

# import transformers
dt = {}
dt['max_lines'] = 0
dt['_lines'] = 0

from vocab import Vocabulary
from multihead_model import MultiHeadModel
from utils import Accuracy, Recall, tokenize, logger

def tokens_to_ids(instance, vocab, config):

    return {'token_id_list': [x for tokens in instance['token_list'] for x in vocab.map_tokens_to_ids(tokens, max_length=config.MAX_LENGTH)]}

def re_organize(instance):
    """Simple wrapper that applies the `tokenize` function to an instance."""
    text = instance['context']
    parts = text.split('.')
    quotes = instance['text']

    _list = []
    token_list = []
    labels = []
    dt['_lines'] +=len(quotes)

    for quote in quotes:
        _list = _list + quote.split('.')
            
    for sent in parts:
        if sent in _list:
            labels.append(1)
            dt['max_lines']+=1
        else:
            labels.append(0)
    new_labels = []

    for i, part in enumerate(parts):
        if len(part) < 35 and len(token_list) >0 and labels[i] !=1:
            token_list[-1] = token_list[-1]+ tokenize(part)
            new_labels.append(0)
        else:
            token_list.append(tokenize(part))
            new_labels.append(labels[i])

    for i in range(len(token_list), 32):
        token_list.append([])
        new_labels.append(0)

    return {'token_list': token_list[:32], 'label': new_labels[:32] }


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
    print(accuracy.total, recall.total, recall.total/accuracy.total)

def validate(val_dataloader, model, config, best_val_recall ):
    logger.info('Validate')
    model.eval() # THIS PART IS VERY IMPORTANT TO SET BEFORE EVALUATION
    accuracy = Accuracy()
    recall = Recall()


    progress_bar = tqdm(val_dataloader)
    for batch in progress_bar:

        # Move data onto GPU.
        batch = {k: v.to(device=config.DEVICE) for k, v in batch.items()}

        # Compute the model output.
        output_dict = model(**batch)

        # Compute accuracy for monitoring
        accuracy.update(output_dict['predictions'], batch['label'])
        recall.update(output_dict['predictions'],batch['label'])
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
    dataset = prepare_qa_dataset()
    train_data, val_data, test_data = dataset["train"], dataset["valid"], dataset["test"]
    
    logger.info("features: " + str(train_data.info.features))

    train_data = train_data.map(re_organize)
    print("*******",dt['max_lines'], dt['_lines'])
    val_data = val_data.map(re_organize)
    test_data = test_data.map(re_organize)


    # Count the token frequencies.
    counter = collections.Counter()
    for instance in train_data:
        counter.update([el for token_list in instance['token_list'] for el in token_list])

    top_words = [x[0] for x in counter.most_common(config.VOCAB_SIZE)]

    vocab = Vocabulary(top_words, add_unk_token=True)

    train_data = train_data.map(tokens_to_ids, fn_kwargs={'vocab':vocab, 'config': config})
    val_data = val_data.map(tokens_to_ids, fn_kwargs={'vocab':vocab, 'config': config})
    test_data = test_data.map(tokens_to_ids, fn_kwargs={'vocab':vocab, 'config': config})

    train_data.set_format(type='torch', columns=['id', 'token_id_list', 'label'])
    val_data.set_format(type='torch', columns=['id', 'token_id_list', 'label'])
    test_data.set_format(type='torch', columns=['id', 'token_id_list', 'label'])

    return train_data, val_data, test_data, vocab

def main():

    class Config:pass
    config = Config()

    config.VOCAB_SIZE = 20000  # No magic numbers.
    config.MAX_LENGTH = 25
    config.EPOCHS = 20
    config.BATCH_SIZE = 1
    config.DROPOUT_RATE = 0.1
    config.HIDDEN_DIM = 192
    config.LEARNING_RATE = 3e-4
    config.DEVICE = 'cuda' if  torch.cuda.is_available() else 'cpu'
    logger.info(f"Running on {config.DEVICE}")


    train_data, val_data, test_data, vocab = prepare_data(config)


    vocab_size = len(vocab)
    with open('checkpoints/attention_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    model = MultiHeadModel(
        vocab_size=vocab_size,
        n_classes=2,
        hidden_dim=config.HIDDEN_DIM,
        dropout_rate=config.DROPOUT_RATE
    )
    model.to(device=config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE)


    best_val_recall = 0.0
    for epoch in range(config.EPOCHS):
        logger.info('\nEpoch' + str(epoch))
        train(train_dataloader, model, optimizer, config)
        best_val_recall  = validate(val_dataloader, model, config, best_val_recall)



if __name__ == '__main__':
    main()
