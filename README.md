# ACE - Boosting Certified Robustness via a Compositional Network architecture

## Setup
Create a new virtualenv with python 3.8 and install the requirements by running
```
git clone https://github.com/eth-sri/ACE
cd ACE
virtualenv --python /usr/bin/python3.8 env_ace
source ./env_ace/bin/activate
pip install -r requirements.txt
cd ..
git clone https://github.com/KaidiXu/auto_LiRPA
cd auto_LiRPA
python setup.py install
```

To conduct any training or evaluation of ImageNet set `IMAGENET_DIR` in `loaders.py`to your local ImageNet directory.
To download the TinyImageNet datastet navigate to your local ImageNet folder specified in `loaders.py` and run the `tinyimagenet_download` located in `data`:
```
bash tinyimagenet_download.sh
```

Our pretrained models can be downloaded [here](https://files.sri.inf.ethz.ch/ace/trained_models.zip).

## Evaluating ACE-Models
ACE-Models can be evaluated jointly using the `evaluate_ACE` script or the core-network can be evaluated separately from the other and the results combined the results using  the `aggreagte_ACE` script.

Ensure in both cases that the right relaxation type is selected. For COLT trained models use `--cert-domain zono` and for IBP trained models use `--cert-domain box`.

Per default the model will be evaluated on the test (or in the case of imagenet validation) set. To evaluate on the train set, set `--test-set train`

Per default an attempt to certify the core-network will be made. 
As the core-networks used in this work are not provably trained and typically quite larger, certification is usually neither feasible nor successful.
Therefore, we recommend turning it off by setting `--cert-trunk False`

### Joint evaluation
To evaluate an ACE model jointly, the underlying architectures and selection parameters have to be specified and the component networks can either all be loaded from the same checkpoint or from different checkpoints.

To load all models from the same file use the `--load-model <path>` option. 
To load the core-, certification- and selection-networks use `--load-trunk-model <path>`, `--load-branch-model <path>`, and `--load-gate-model <path>`, respectively.

### Separate evaluation
To evaluate the components of an ACE model separately, the core-network has to be evaluated using `python main.py --train-mode test` and the certification block consisting out of the certification-network and selection mechanism has to be evaluated as for joint evaluation, but specifying `--net None`.
This generates a `test_log.csv` and `cert_log.csv` file, which have to be combined using `aggregate_ACE`.

The resulting natural and certified accuracy are identical to what you would obtain with joint evaluation.
The empiricial PGD accuracies are two approximate lower bounds, one computed using empirical reachability, the other one using certified reachability.
Typically both of these bounds underestimate the empirical adversarial accuracy notably.

## Training ACE-Models
While all component networks can be trained consecutively using `deepTrunk_main.py`, we suggest training the core- and certification-network in advance.
Further we propose to initialize the selection-network, where applicable, with the provably trained certification network.
When using Entropy Selection we propose the same approach even though the certification network will be retrained using the entropy loss component.
This leads to the follwoing steps to train an ACE-Model from scratch:
1. Use `main.py` to train the core-network using `--train-mode train-adv`
2. Pretrain the certification network using `main.py`. 
    If a multi resolution ACE-Model is to be trained, load the data in the highest resolution and specify the certification net resolution as `--cert-net-dim`
3. Train the selection network (for Selection ACE) or retrain the certificaiton-network (for an Entropy ACE), using `deepTrunk_main.py` and setting `--net None`
 
 ### Selection Mechanism
 The selection mechanism is specified by `--gate-type` with the two valid options being `net`for a SelectionNet type selection and `entropy` for an entropy selection.
 
 The selection threshold for both methods is set using `--gate-threshold`.
 
 #### Entropy Selection
 If entropy is specified all parameters specifying the gate net will be ignored.
 
 To activate joint training set `--cotrain-entropy 0.5` to the desired weighting factor, typically 0.5.
  
 Entropy thresholds have to be spcified negated: To specify a maximum entropy for selection of 0.4, `--gate-threshold -0.4` hast to be set.
 
 #### SelectionNet
 Specify the selection-network architecture with `--gate-nets` or don't set parameter to use the same as for the certificaiton-network.
 
 Setting `--gate-feature-extraction 1` will only retrain the last layer of the selection network.
 
 ### Provable Training Type
 To train using COLT set `--train-mode train-COLT`.\
 For IBP training set `--train-mode train-diffAI --train-domain box`.\
 For different CROWN variation use `--train-mode train-diffAI` und see below for the supported domains.

 #### COLT
 COLT offers quite a few options to be set, specifying the exact behaviour over the different, attacked latent-spaces.
 The most important ones are listed below. A full list can be found in `args_factory.py` including short descriptions.
 * `--n-attack-layers` setting the maximum number of layers to be attacked
 * `--protected-layers` setting how many layers from the last one not to attack
 * `--relu-stable` setting the weight of the ReLU stabilization loss
 * `--eps-factor` factor to (cumulatively) increase early layer \epsilon 
 
 #### IBP
 For IBP training the most important settings are
 * `--nat-factor` (equivalent to \kappa_{end}), defining the weight of natural and robust loss.
 * `--mix` and `--mix-epochs` which effectively define a nat-factor annealing schedule from 1 to the specified value
 * `--anneal` and `--anneal-epochs` defining an \epsilon annealing

 #### Auto LiRPA
 The IBP, CROWN, and CROWN-IBP can be accessed using `--train-domain {l_IBP, l_CROWN, l_CROWN-IBP}` without loss-fusion or `{lf_IBP, lf_CROWN, lf_CROWN-IBP}` 
 with loss-fusion. For more details on Auto LiRPA see [Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond](https://arxiv.org/pdf/2002.12920).

## Publications
* [Certify or Predict: Boosting Certified Robustness with Compositional Architectures](https://openreview.net/pdf?id=USCNapootw) \
  Mark Niklas Mueller, Mislav Balunovic, Martin Vechev\
  ICLR2021
  