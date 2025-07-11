import torch
import torch.nn as nn
from typing import List
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> Tensor:
        # Step 1: Collect all unique words
        vocabulary = set()
        for sentence in positive + negative:
            for word in sentence.split():
                vocabulary.add(word)
        
        # Step 2: Sort vocabulary lexicographically and map to integers starting from 1
        sorted_vocab = sorted(vocabulary)
        word_to_index = {word: idx + 1 for idx, word in enumerate(sorted_vocab)}  

        # Step 3: Encode and store positive and negative sequences separately
        encoded_positive = []
        encoded_negative = []

        for sentence in positive:
            tokens = [word_to_index[word] for word in sentence.split()]
            encoded_positive.append(torch.tensor(tokens, dtype=torch.float))

        for sentence in negative:
            tokens = [word_to_index[word] for word in sentence.split()]
            encoded_negative.append(torch.tensor(tokens, dtype=torch.float))

        # Step 4: Combine encoded lists in the correct order: positive first, then negative
        all_encoded = encoded_positive + encoded_negative

        # Step 5: Pad all sequences to the length of the longest one 
        padded_tensor = pad_sequence(all_encoded, batch_first=True)

        return padded_tensor
