# JetCLR
Code base for the JetCLR tool, providing a framework for the contrastive learning of jet observables in high-energy particle phenomenology.

***

## Approximate pipeline

### Data Preprocessing

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

