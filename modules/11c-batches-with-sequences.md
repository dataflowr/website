@def sequence = ["batches-seq"]

# Module 11c - Batches with sequences in Pytorch


**Table of Contents**

\toc


## Pytorch tutorial on batch for sequences

{{youtube_placeholder batches-seq}}
{{yt_tsp 0 0 Presentation}}
{{yt_tsp 135 0 Step 1: Construct Vocabulary}}
{{yt_tsp 170 0 Step 2: Load indexed data (list of instances, where each instance is list of character indices)}}
{{yt_tsp 225 0 Step 3: Make Model}}
{{yt_tsp 290 0 Step 4: Pad instances with 0s till max length sequence}}
{{yt_tsp 347 0 Step 5: Sort instances by sequence length in descending order}}
{{yt_tsp 415 0 Step 6: Embed the instances}}
{{yt_tsp 550 0 Step 7: Call pack_padded_sequence with embeded instances and sequence lengths}}
{{yt_tsp 761 0 Step 8: Forward with LSTM}}
{{yt_tsp 878 0 Step 9: Call unpack_padded_sequences if required / or just pick last hidden vector}}

## Notebook

- [notebook](https://github.com/dataflowr/notebooks/blob/master/Module11/11_Tutorial_packing_sequences.ipynb)