# JetCLR
Code base for the JetCLR tool, providing a framework for the contrastive learning of jet observables in high-energy particle phenomenology.

JetCLR uses a permutation-invariant transformer-encoder network and a contrastive loss function to map jet constituents to a representation space which is approximately invariant to a set of symmetries and augmentations of the jet data, and is discriminant within the dataset it is optimised on. The symmetries and augmentations are coded in scripts/modules/jet_augs.py, they are:

- **symmetries**:
  - jet-wide rotations in the rapidity-azimuthal plane, around the transverse-momentum-weighted centroid of each jet
  - event-wide translations in the rapidity-azimuthal plane
  - permutation invariance of the jet constituents, this is ensured by the network architecture
- **augmentations**:
  - smearing of constituent coordinates, inversely proportional to their transverse momentum
  - collinear splittings of jet constituents

The scheme for optimising the network is inspired by the [SimCLR1 paper](https://arxiv.org/abs/2002.05709). The mapping to the new representation space is entirely self-supervised, using only the physically-motivated invariances to transformations and augmentations of the data. Truth labels are not used in the optimisation of the JetCLR network.

For questions/comments about the code contact: rmastand@berkeley.edu or dillon@thphys.uni-heidelberg.de

---

This code was written for the paper: 

Self-supervised Anomaly Detection for New Physics  
https://arxiv.org/abs/2205.10380   
Barry M. Dillon, Radha Mastandrea, and Benjamin Nachman  


The [first iteration of this repository](https://github.com/bmdillon/JetCLR) was written for the paper:

Symmetries, Safety, and Self-Supervision  
https://arxiv.org/abs/2108.04253   
Barry M. Dillon, Gregor Kasieczka, Hans Olischlager, Tilman Plehn, Peter Sorrenson, and Lorenz Vogel  

***

## Approximate pipeline

### Preprocessing the data 

```data_prep.py```: Processes the LHC Olympics dataset into a more compact form. Creates a text file of clustered jets.

```data_preprocessing.ipynb```: Makes selection cuts on the proccessed data and creates datasets with a setable number of signal and background events

```aug_testing.ipynb```: Allows for visualization of jet augmentations in the $\eta$-$\phi$ plane. Also shows the effect of augmentations on jet observables $\tau$, $m$, ...

### Training the transformer networks

```CLR_training_transformer.py```: trains a transformer network to map events into a latent space using the contrastive loss. Also trains a binary classifier (BC) to discriminate signal from background events in the latent space.

```BC_training_transformer.py```: trains a transformer network to directly do a SvB classifiction task in a latent space with the BCE Loss (the "Trans+BC" network).

```event_space_BC.ipynb```: runs the benchmark binary classfication tasks (fully connected network + linear classifier test) in the event space.

### Visualizing the transformer performances

```dim_scan_plots.ipynb```: plots rejection vs TPR for the transformer trained on a given latent space dimension.

```SvB_plots.ipynb```: plots rejection vs TPR for the transformer trained on with a given S/B value. Also plots training metrics (ROC AUC, max(SIC), and FPR at TPR = 0.5) as a function of training epoch.

### Running a weakly-supervised (CWoLa) anomaly detection search

```JetCLR_CWoLa.py```: runs a CWoLA anomaly detection procedure in the latent space

```JetCLR_CWoLa_event_space```: runs a CWoLA anomaly detection procedure in the event space

```CWoLa_plots.ipynb```: plots training metrics (ROC AUC, max(SIC), and FPR at TPR = 0.5) as a function of the anomaly / signal fraction

### Sanity Checks

```sanity_checks.ipynb```: explores the performance gap between anomaly detection in a latent space vs. SOTA methods

