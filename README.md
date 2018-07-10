# Segmental DAM pytorch implementation
This project contains code, hyper-parameters and serialized models for the open source implementation of different BoW attention models.
The full description is available here: TBA

If you use this software for academic research please cite the described paper.

# Requirements
- Python 2
- Python 3
- Pytorch (tested on 0.2)

# Usage
Download the datasets and run the preprocessing scripts (Preprocess_scripts folder):

```
path=..../path_to_stsbenchmark_folder
glove=..../path_to_glove_folder


python3 Preprocess_scripts/process-STSBenchmark.py 
    --data_folder ${path}
    --out_folder ${path}

for name in glove.6B.300d glove.42B.300d glove.840B.300d
do
    python2 Preprocess_scripts/preprocess-STSBenchmark.py
	--srcfile ${path}/src-train.txt
	--targetfile ${path}/targ-train.txt
	--labelfile ${path}/label-train.txt
	--srcvalfile ${path}/src-dev.txt
	--targetvalfile ${path}/targ-dev.txt
	--labelvalfile ${path}/label-dev.txt
	--srctestfile ${path}/src-test.txt
	--targettestfile ${path}/targ-test.txt
	--labeltestfile ${path}/label-test.txt
	--batchsize 4
	--outputfile ${path}/$name
	--glove ${glove}/${name}.txt
done

for name in glove.6B.300d glove.42B.300d glove.840B.300d
do
    python2 Preprocess_scripts/get_pretrain_vecs.py
	--glove ${glove}/${name}.txt
	--outputfile ${path}/${name}.data.hdf5
	--dictionary ${path}/${name}.word.dict
done

```

Launch the models:

```
usage: DAM_STSBenchmark_TS_Segmental_MaxSpan.py [-h] 
    [--train_file TRAIN_FILE]
    [--dev_file DEV_FILE]
    [--test_file TEST_FILE]
    [--bigrams]
    [--trigrams]
    [--dropout DROPOUT]
    [--w2v_file W2V_FILE]
    [--embedding_size EMBEDDING_SIZE]
    [--log_dir LOG_DIR]
    [--log_fname LOG_FNAME]
    [--gpu_id GPU_ID]
    [--epoch EPOCH]
    [--dev_interval DEV_INTERVAL]
    [--optimizer OPTIMIZER]
    [--Adagrad_init ADAGRAD_INIT]
    [--lr LR]
    [--hidden_size HIDDEN_SIZE]
    [--max_length MAX_LENGTH]
    [--maxSpan MAXSPAN]
    [--display_interval DISPLAY_INTERVAL]
    [--max_grad_norm MAX_GRAD_NORM]
    [--para_init PARA_INIT]
    [--weight_decay WEIGHT_DECAY]
    [--model_path MODEL_PATH]
    [--seed SEED]

```
# Demo
There is a demo file inside the code directory to show how models are loaded to replicate the results on the SNLI dataset, find below a small snippet of the file:

```
from models.baseline_snli import encoder
from models.baseline_snli import atten

....

torch.cuda.set_device(args.gpu_id)
test_data = snli_data( ..../Path_to_snli, 9999)
test_batches = test_data.batches
test_lbl_size = 3

....

correct = 0.
total = 0.
for i in range(len(test_batches)):
    test_src_batch, test_tgt_batch, test_lbl_batch = test_batches[i]
    test_src_batch = Variable(test_src_batch.cuda())
    test_tgt_batch = Variable(test_tgt_batch.cuda())
    test_lbl_batch = Variable(test_lbl_batch.cuda())
    test_src_linear, test_tgt_linear = input_encoder(test_src_batch, test_tgt_batch)
    log_prob = inter_atten(test_src_linear, test_tgt_linear)
    _, predict = log_prob.data.max(dim=1)
    total += test_lbl_batch.data.size()[0]
    correct += torch.sum(predict == test_lbl_batch.data)
```

# Serialized models
Models can be downloaded from google drive (total of 1.2G)

- DAM BoW: 
- DAM CNN: 
- DAM REC: 
- DAM Seg: 

# Acknowledgements

The project is motivated by the following paper and github repositories:

* A. Parikh, O. Täckström, D. Das, J. Uszkoreit, A decomposable attention model for natural language inference, in: Proceedings of the 2016 Conference on Empirical Methods in Natural Language Process
ing, Association for Computational Linguistics, Austin, Texas, 2016, pp. 2249–2255. URL https://aclweb.org/anthology/D16-1244
* Decomposable Attention Model for Sentence Pair Classification, https://github.com/harvardnlp/decomp-attn
* SNLI-decomposable-attention, https://github.com/libowen2121/SNLI-decomposable-attention
* decomposable_attention, https://github.com/shuuki4/decomposable_attention
