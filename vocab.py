class Vocabulary():
    """ Object holding vocabulary and mappings
    Args:
        word_list: ``list`` A list of words. Words assumed to be unique.
        add_unk_token: ``bool` Whether to create an token for unknown tokens.
    """
    def __init__(self, word_list, add_unk_token=False):
        # create special tokens for padding and unknown words
        self.pad_token = '<pad>'
        self.unk_token = '<unk>' if add_unk_token else None

        self.special_tokens = [self.pad_token]
        if self.unk_token:
            self.special_tokens += [self.unk_token]

        self.word_list = word_list
        
        # maps from the token ID to the token
        self.id_to_token = self.special_tokens + self.word_list
        # maps from the token to its token ID
        self.token_to_id = {token: id for id, token in
                            enumerate(self.id_to_token)}
        
    def __len__(self):
        """ Returns size of vocabulary """
        return len(self.token_to_id)
    
    @property
    def pad_token_id(self):
        return self.map_token_to_id(self.pad_token)
        
    def map_token_to_id(self, token: str):
        """ Maps a single token to its token ID """
        if token not in self.token_to_id:
            token = self.unk_token
        return self.token_to_id[token]

    def map_id_to_token(self, id: int):
        """ Maps a single token ID to its token """
        return self.id_to_token[id]

    def map_tokens_to_ids(self, tokens: list, max_length: int = None):
        """ Maps a list of tokens to a list of token IDs """
        # truncate extra tokens and pad to `max_length`
        if max_length:
            tokens = tokens[:max_length]
            tokens = tokens + [self.pad_token]*(max_length-len(tokens))
        return [self.map_token_to_id(token) for token in tokens]
    
    def tokens_to_ids(self, instance):
        return {'token_ids': self.map_tokens_to_ids(instance['tokens'], max_length=config.MAX_LENGTH)}


    def map_ids_to_tokens(self, ids: list, filter_padding=True):
        """ Maps a list of token IDs to a list of token """
        tokens = [self.map_id_to_token(id) for id in ids]
        if filter_padding:
            tokens = [t for t in tokens if t != self.pad_token]
        return tokens
