
Drug-target binding affinity prediction through complementary biological-related and compression-based featurization approach

## Data
We utilized four DTA datasets including Davis, Kiba refine set. Davis and Kiba datasets were downloaded from [here](https://github.com/hkmztrk/DeepDTA/tree/master/data). 
<br/>
Each dataset folder includes binding affinity (i.e. Y), protein sequences (i.e. proteins.txt), drug SMILES (i.e. ligands_can.txt, ligands_iso.txt), and encoded protein sequences (i.e. protVecNCD  and protVecSW) files, and a folder includes the train and test folds settings (i.e. folds).

## Requirements
Python <br/>
Tensorflow <br/>
Keras <br/>Z
Numpy <br/>

  
## Usage
For training and evaluation of the method, you can run the following script.

```
python run_experiments.py --num_windows 128 8 --smi_window_lengths 4 8 16 --batch_size 512 --num_epoch 1000 --max_smi_len 85 --dataset_path data/kiba/ --problem_type 1 --log_dir logs/ 
```


## Cold-start
protein cold-start change value of problem_type to 2, drug cold-start change value of problem_type to 3 , protein-drug cold-start change value of problem_type to 4. 

```
python run_experiments.py --num_windows 128 8 --smi_window_lengths 4 8 16 --batch_size 512 --num_epoch 1000 --max_smi_len 85 --dataset_path data/davis/ --problem_type 3 --log_dir logs/ 
```

