# JetCLR
Code base for the JetCLR tool, providing a framework for the contrastive learning of jet observables in high-energy particle phenomenology.


Approximate pipeline:

```data_prep.py```: Processes the LHC Olympics dataset into a more compact form. Creates a text file of clustered jets.

```data_preprocess.ipynb```: Makes selection cuts on the proccessed data and creates datasets with a setable number of signal and background events

```aug_testing.ipynb```: Allows for visualization of jet augmentations in the $\eta$-$\phi$ plane. Also shows the effect of augmentations on jet observables $\tau$, $m$, ...

```CLR_training_transformer.py```: trains a transformer network to map events into a latent space using the contrastive loss. Also trains a binary classifier (BC) to discriminate signal from background events in the latent space.

```BC_training_transformer.py```: trains a transformer network to directly do a SvB classifiction task in a latent space with the BCE Loss (the "Trans+BC" network).

```event_space_BC.ipynb```: runs the benchmark binary classfication tasks (fully connected network + linear classifier test) in the event space.
