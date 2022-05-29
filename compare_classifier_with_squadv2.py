import pandas as pd
from utils import Accuracy, Recall, tokenize, tokenize_instance, logger
import torch
from datasets import Dataset, load_metric
import pickle
from model import Model
from tqdm import tqdm


def convert_pred_to_df(predictions):
    df = pd.DataFrame({'id':[], 'text':[], 'label':[]})
    for ex in predictions:
        df.loc[len(df.index)] = [ex['id'], ex['prediction_text'], 1]
    return df, 2

def tokens_to_ids(instance, vocab):
    return {'token_ids': vocab.map_tokens_to_ids(instance['tokens'], max_length=60)}


def validate(val_dataloader, model ):
    logger.info('Validate')
    model.eval() 
    no_ans_probs={}

    DEVICE = 'cuda' if  torch.cuda.is_available() else 'cpu'
    progress_bar = tqdm(val_dataloader)
    for batch in progress_bar:
        batch = {k: v.to(device=DEVICE) for k, v in batch.items()}
        output_dict = model(**batch)
        no_ans_probs.update({batch['id'][idx].item() : output_dict['no_answer_probability'][idx] for idx in range(len(batch['label'])) })
    return no_ans_probs


def compare_with_classifier(predictions, references):
    vocab = None
    with open('checkpoints/vocab.pkl', 'rb') as f:
        vocab =pickle.load(f)

    state_dict = torch.load('checkpoints/best-model.pt')
    df, num_classes = convert_pred_to_df(predictions)

    model = Model(
        vocab_size=len(vocab),
        n_classes=num_classes,
        hidden_dim=128,
        dropout_rate=0.1
    )   
    model.load_state_dict(state_dict)

    val_data = Dataset.from_pandas(df)

    val_data = val_data.map(tokenize_instance)

    val_data = val_data.map(tokens_to_ids, fn_kwargs={'vocab':vocab})

    val_data.set_format(type='torch', columns=['id', 'token_ids', 'label'])
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=16)
    no_ans_probs = validate(val_dataloader, model)

    metric = load_metric("squad_v2")

    print("compare mode 1")
    new_predictions = [{'id': ex['id'], 'prediction_text': ex['prediction_text'] if no_ans_probs[ex['id']] < 0.5 else '', 'no_answer_probability': 0.0 } for ex in predictions]
    print(metric.compute(predictions=new_predictions, references=references))

    print("compare mode 2")
    new_predictions = [{'id': ex['id'], 'prediction_text': ex['prediction_text'], 'no_answer_probability': no_ans_probs[ex['id']]  } for ex in predictions]
    print(metric.compute(predictions=new_predictions, references=references))
    

if __name__ == '__main__':
    predictions = [{'id': 0, 'prediction_text': 'sample', 'no_answer_probability': 0 }, {'id': 1, 'prediction_text': 'something', 'no_answer_probability': 0 }]
    references = [{'id': 0, "answers": {'answer_start': [0], 'text': ['sample']} }, {'id': 1, "answers": {'answer_start': [], 'text': []} } ]
    compare_with_classifier(predictions, references)
