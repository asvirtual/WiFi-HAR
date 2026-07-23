# Contrastive Learning Report

This document summarizes the contrastive-learning experiment we ran on top of the Wi-Fi Doppler activity classifier, the results we observed, and the most reasonable next directions.

### What We Tried

We started from the convolutional baseline and added a supervised contrastive learning objective in order to make the latent space more discriminative. The encoder was kept the same, while a projection head was added only for the contrastive branch.

At first, we used augmented versions of the same spectrogram window as positive pairs, following the same masking strategy already used in the baseline. The goal was to make the model learn invariances while still predicting the activity label with cross-entropy.

Later, after noticing that the masking was probably too aggressive, we switched to a cleaner formulation where the four antenna recordings of the same event were used as aligned positive views. This was a more meaningful source of positives because the antennas observe the same action from slightly different perspectives.

The final training objective was a weighted sum of:

- label-smoothed cross-entropy for classification;
- supervised contrastive loss for representation learning.

We also introduced a warm-up for the contrastive term so that the classification part could stabilize first.

### What The Training Showed

The training curves were initially encouraging. During the best run, the model reached roughly 0.72 validation accuracy at its peak, while the validation loss dropped to around 1.0 at best. This looked promising at first glance, because it suggested that the encoder was learning something useful and that the contrastive branch was not completely destabilizing training.

However, after looking at the evaluation more carefully, the result was not as strong as the raw accuracy number suggested.

The final evaluation of the second contrastive model gave:

- Accuracy: 58.96%
- Macro F1-score: 57.25%

The best validation loss during training was about 1.25, and the best validation accuracy was around 61.2%, but the confusion matrix showed that the model was still mixing several difficult classes. In other words, the model was not truly separating the activities well; it was often getting the right global score while still confusing the important classes.

### What Did Not Work Well

The first issue was the masking. Strong time and frequency masking made the task harder than it should have been, especially when the same masked views were also used for the contrastive branch. In practice, this likely destroyed some of the spectral structure that the model needed to preserve.

The second issue was that the contrastive objective was too difficult too early in training. The training loss remained noticeably higher than the validation loss, which is not automatically a bug in this setting, but it does indicate that the model was seeing a harder optimization problem during training than during evaluation.

The third issue was that the confusion matrix did not match the improvement suggested by the accuracy curve. The model still confused the more similar activities and the lower-energy classes, which means the representation was not yet robust enough for fine-grained recognition.

So although the aggregate numbers were not terrible, the model was still underperforming in the sense that it did not produce a clean class separation.

### Best Conclusions So Far

The most important conclusion is that contrastive learning does help to structure the embedding space, but only if the positive pairs are meaningful.

The four antennas are a better source of positives than synthetic masking alone, because they correspond to the same real event and therefore preserve the underlying activity label more faithfully.

Another conclusion is that the model likely benefits more from stable, semantically valid views than from heavy augmentation. For this dataset, too much masking seems to remove signal instead of removing nuisance.

Finally, the contrastive branch should probably be treated as a regularizer or a pretraining signal, not as the sole thing driving the whole training process from the first epoch.

### Brainstorming: What Should We Try Next?

The next iteration should focus on making the positive pairs cleaner and the optimization less noisy. The most sensible directions are:

- pretrain the encoder with supervised contrastive learning on the four antenna views, then fine-tune with cross-entropy only;
- reduce masking even further, or remove it entirely from the contrastive branch;
- keep the clean spectrogram for classification and use the antenna views only for the contrastive objective;
- test a stronger multi-view fusion strategy, instead of just averaging the logits across antennas;
- move the contrastive experiment onto the stronger recurrent or late-fusion backbone, since temporal modeling already helped in the non-contrastive experiments;
- build more balanced batches so that supervised contrastive loss sees enough same-class positives in each batch;
- tune the contrastive temperature and projection head size only after the view strategy is stable.

### Short Summary

The contrastive experiment was not a failure, but it was also not the final answer. It improved the representation somewhat, and the antenna-based positive pairs are a better idea than masking alone, but the confusion matrix shows that the model still lacks the class separation we need. The next step should therefore be to simplify the augmentation, use the antennas more carefully, and probably combine contrastive learning with a stronger temporal backbone.

### Pretraining Version: What We Tried

In the second contrastive version, we changed the training scheme from a single mixed objective to a two-stage pipeline. First, we pre-trained the encoder with supervised contrastive learning on the four antenna views of the same event. In this stage, the backbone and projection head were optimized to bring embeddings of the same activity closer together, while the classifier was kept out of the objective.

After that, we fine-tuned the model with cross-entropy only. The idea was to let the encoder learn a better activity representation first, and then let the classifier specialize on top of those features. This was meant to reduce the instability we saw when the contrastive and classification losses were optimized together from the beginning.

### Pretraining Version: What The History Shows

The pretraining loss decreased slowly but did not show a strong separation between train and validation behavior. In fact, the validation contrastive loss stayed high and noisy across epochs, which suggests that the encoder was learning useful structure only partially. The best encoder checkpoint was saved early, but the later pretraining epochs did not visibly improve the situation.

The fine-tuning stage was more interesting. The model reached peaks around 0.72 to 0.75 validation accuracy and even hit a best value close to 0.76 at one point, which is better than the earlier contrastive version. At the same time, the validation loss remained unstable, and the model repeatedly moved between better and worse epochs. The early stopping at the end also confirms that the training was not converging in a clean way.

### Conclusions From The Pretraining Run

The main conclusion is that pretraining the encoder was the right idea, but it was not enough by itself to solve the classification problem. The staged setup was more sensible than mixing everything from the first epoch, because it gave the model a clearer optimization path. Even so, the final behavior still shows that the representation is not fully robust for all classes.

More specifically, the experiment suggests that:

- the four antenna views are still a better positive signal than masking;
- the encoder benefits from contrastive pretraining, but the gain is limited if the backbone is not strong enough;
- fine-tuning improves the validation accuracy, but the confusion matrix likely remains the real bottleneck;
- the training is still noisy enough that a stronger backbone or a better multi-view fusion strategy may be needed next.

So the pretraining version is a useful step forward, but not the final solution. It confirms that staged training is preferable, while also showing that the model architecture itself still needs improvement if we want cleaner class separation.



-------------
# CONTRASTIVE WITH PRETRAINING


[PRETRAIN] Epoch: 1

  0%|          | 0/919 [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/nn/modules/conv.py:548: UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created (Triggered internally at /pytorch/aten/src/ATen/native/Convolution.cpp:1025.)
  return F.conv2d(
Pretrain loss: 4.10690: 100%|██████████| 919/919 [02:22<00:00,  6.47it/s]
Pretrain val loss: 4.12110: 100%|██████████| 226/226 [00:15<00:00, 14.80it/s]

[PRETRAIN] train loss: 4.782920219357012, val loss: 5.723230281952889
[PRETRAIN] Saved encoder checkpoint
[PRETRAIN] Epoch: 2

Pretrain loss: 4.02780: 100%|██████████| 919/919 [02:22<00:00,  6.43it/s]
Pretrain val loss: 4.12218: 100%|██████████| 226/226 [00:15<00:00, 14.82it/s]

[PRETRAIN] train loss: 4.650530843677635, val loss: 5.7408938828665805
[PRETRAIN] Epoch: 3

Pretrain loss: 3.88248: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Pretrain val loss: 4.19976: 100%|██████████| 226/226 [00:15<00:00, 14.77it/s]

[PRETRAIN] train loss: 4.583233678736762, val loss: 5.775852374918048
[PRETRAIN] Epoch: 4

Pretrain loss: 3.80723: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Pretrain val loss: 4.21223: 100%|██████████| 226/226 [00:15<00:00, 14.79it/s]

[PRETRAIN] train loss: 4.50607315690607, val loss: 5.750290131213307
[PRETRAIN] Epoch: 5

Pretrain loss: 3.73706: 100%|██████████| 919/919 [02:23<00:00,  6.41it/s]
Pretrain val loss: 4.26110: 100%|██████████| 226/226 [00:15<00:00, 14.84it/s]

[PRETRAIN] train loss: 4.4238644432748995, val loss: 5.8162408233639935
[PRETRAIN] Epoch: 6

Pretrain loss: 3.87835: 100%|██████████| 919/919 [02:23<00:00,  6.41it/s]
Pretrain val loss: 4.24758: 100%|██████████| 226/226 [00:15<00:00, 14.87it/s]

[PRETRAIN] train loss: 4.3795919600036655, val loss: 5.774118406459188
[PRETRAIN] Epoch: 7

Pretrain loss: 3.78915: 100%|██████████| 919/919 [02:23<00:00,  6.41it/s]
Pretrain val loss: 4.26277: 100%|██████████| 226/226 [00:15<00:00, 14.89it/s]

[PRETRAIN] train loss: 4.323478784519625, val loss: 5.8440578307867295
[PRETRAIN] Epoch: 8

Pretrain loss: 3.61306: 100%|██████████| 919/919 [02:23<00:00,  6.39it/s]
Pretrain val loss: 4.35312: 100%|██████████| 226/226 [00:15<00:00, 14.87it/s]

[PRETRAIN] train loss: 4.277111963013559, val loss: 5.860322073057548
[FINETUNE] Epoch: 1

Finetune loss: 0.85794: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Finetune val loss: 0.61754: 100%|██████████| 226/226 [00:15<00:00, 14.72it/s]

[FINETUNE] train loss: 0.9700073074502794, val loss: 1.1105956439173061, accuracy: 0.7216094346167187
[FINETUNE] Saved model
[FINETUNE] Epoch: 2

Finetune loss: 0.81659: 100%|██████████| 919/919 [02:23<00:00,  6.40it/s]
Finetune val loss: 1.14170: 100%|██████████| 226/226 [00:15<00:00, 14.67it/s]

[FINETUNE] train loss: 0.8413433532190504, val loss: 1.235492820141012, accuracy: 0.6237252861602497
[FINETUNE] Epoch: 3

Finetune loss: 0.89317: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Finetune val loss: 2.05964: 100%|██████████| 226/226 [00:15<00:00, 14.67it/s]

[FINETUNE] train loss: 0.8102931453277056, val loss: 1.2754019807040835, accuracy: 0.5367325702393341
[FINETUNE] Epoch: 4

Finetune loss: 0.70428: 100%|██████████| 919/919 [02:23<00:00,  6.41it/s]
Finetune val loss: 0.75082: 100%|██████████| 226/226 [00:15<00:00, 14.77it/s]

[FINETUNE] train loss: 0.7928728904812377, val loss: 1.2673707360779705, accuracy: 0.642386403052376
[FINETUNE] Epoch: 5

Finetune loss: 0.77434: 100%|██████████| 919/919 [02:23<00:00,  6.43it/s]
Finetune val loss: 0.86040: 100%|██████████| 226/226 [00:15<00:00, 14.71it/s]

[FINETUNE] train loss: 0.7876083343393724, val loss: 1.030689162601004, accuracy: 0.7424210891432536
[FINETUNE] Saved model
[FINETUNE] Epoch: 6

Finetune loss: 0.69453: 100%|██████████| 919/919 [02:23<00:00,  6.41it/s]
Finetune val loss: 0.56943: 100%|██████████| 226/226 [00:15<00:00, 14.87it/s]

[FINETUNE] train loss: 0.770810314281258, val loss: 1.0629674883464373, accuracy: 0.7388831078737427
[FINETUNE] Epoch: 7

Finetune loss: 0.89363: 100%|██████████| 919/919 [02:23<00:00,  6.41it/s]
Finetune val loss: 1.79438: 100%|██████████| 226/226 [00:15<00:00, 14.84it/s]

[FINETUNE] train loss: 0.7673658837945032, val loss: 1.7095915764601248, accuracy: 0.5126604231703087
[FINETUNE] Epoch: 8

Finetune loss: 0.81126: 100%|██████████| 919/919 [02:23<00:00,  6.41it/s]
Finetune val loss: 0.56097: 100%|██████████| 226/226 [00:15<00:00, 14.83it/s]

[FINETUNE] train loss: 0.7642910291010359, val loss: 1.0847972563476642, accuracy: 0.7026708289975719
[FINETUNE] Epoch: 9

Finetune loss: 0.75387: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Finetune val loss: 0.67440: 100%|██████████| 226/226 [00:15<00:00, 14.83it/s]

[FINETUNE] train loss: 0.7513282675423827, val loss: 1.001180160124187, accuracy: 0.7415886229621922
[FINETUNE] Saved model
[FINETUNE] Epoch: 10

Finetune loss: 0.69803: 100%|██████████| 919/919 [02:23<00:00,  6.41it/s]
Finetune val loss: 0.56020: 100%|██████████| 226/226 [00:15<00:00, 14.79it/s]

[FINETUNE] train loss: 0.7377618809854718, val loss: 1.2501040007974635, accuracy: 0.5929240374609781
[FINETUNE] Epoch: 11

Finetune loss: 0.70593: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Finetune val loss: 1.44962: 100%|██████████| 226/226 [00:15<00:00, 14.78it/s]

[FINETUNE] train loss: 0.7331696259436817, val loss: 1.2677854497772598, accuracy: 0.6354491848768644
[FINETUNE] Epoch: 12

Finetune loss: 0.85125: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Finetune val loss: 0.56041: 100%|██████████| 226/226 [00:15<00:00, 14.70it/s]

[FINETUNE] train loss: 0.7106322645816585, val loss: 1.7711125825992111, accuracy: 0.5481096080471731
[FINETUNE] Epoch: 13

Finetune loss: 0.73336: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Finetune val loss: 1.22368: 100%|██████████| 226/226 [00:15<00:00, 14.84it/s]

[FINETUNE] train loss: 0.6963468124247402, val loss: 1.0642539461164313, accuracy: 0.6808185917447104
[FINETUNE] Epoch: 14

Finetune loss: 0.63799: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Finetune val loss: 1.17991: 100%|██████████| 226/226 [00:15<00:00, 14.77it/s]

[FINETUNE] train loss: 0.6851000009701832, val loss: 1.31821041786956, accuracy: 0.5723898716614637
[FINETUNE] Epoch: 15

Finetune loss: 0.63288: 100%|██████████| 919/919 [02:22<00:00,  6.43it/s]
Finetune val loss: 0.99422: 100%|██████████| 226/226 [00:15<00:00, 14.72it/s]

[FINETUNE] train loss: 0.6752509279100546, val loss: 1.190694049634414, accuracy: 0.644120707596254
[FINETUNE] Epoch: 16

Finetune loss: 0.63536: 100%|██████████| 919/919 [02:23<00:00,  6.42it/s]
Finetune val loss: 0.95640: 100%|██████████| 226/226 [00:15<00:00, 14.91it/s]

[FINETUNE] train loss: 0.6619047341167634, val loss: 1.1524728944720886, accuracy: 0.6593132154006244
[FINETUNE] Epoch: 17

Finetune loss: 0.60418: 100%|██████████| 919/919 [02:22<00:00,  6.44it/s]
Finetune val loss: 1.31247: 100%|██████████| 226/226 [00:15<00:00, 14.96it/s]

[FINETUNE] train loss: 0.6486075344329693, val loss: 1.1688153240112558, accuracy: 0.6301075268817204
[FINETUNE] Epoch: 18

Finetune loss: 0.62450: 100%|██████████| 919/919 [02:22<00:00,  6.45it/s]
Finetune val loss: 0.56567: 100%|██████████| 226/226 [00:14<00:00, 15.24it/s]

[FINETUNE] train loss: 0.6356670357445108, val loss: 1.3795260563518619, accuracy: 0.6163718348942074
[FINETUNE] Epoch: 19

Finetune loss: 0.66888: 100%|██████████| 919/919 [02:20<00:00,  6.52it/s]
Finetune val loss: 0.93655: 100%|██████████| 226/226 [00:14<00:00, 15.37it/s]

[FINETUNE] train loss: 0.622079015167498, val loss: 0.991874338147378, accuracy: 0.7553243149497052
[FINETUNE] Saved model
[FINETUNE] Epoch: 20

Finetune loss: 0.71200: 100%|██████████| 919/919 [02:20<00:00,  6.55it/s]
Finetune val loss: 0.79237: 100%|██████████| 226/226 [00:14<00:00, 15.33it/s]

[FINETUNE] train loss: 0.6155910243076933, val loss: 1.3788688799345035, accuracy: 0.6180367672563302
[FINETUNE] Epoch: 21

Finetune loss: 0.60810: 100%|██████████| 919/919 [02:20<00:00,  6.55it/s]
Finetune val loss: 0.93696: 100%|██████████| 226/226 [00:14<00:00, 15.49it/s]

[FINETUNE] train loss: 0.6024182591350038, val loss: 1.1367082069493233, accuracy: 0.679361775927853
[FINETUNE] Epoch: 22

Finetune loss: 0.57927: 100%|██████████| 919/919 [02:20<00:00,  6.56it/s]
Finetune val loss: 0.65719: 100%|██████████| 226/226 [00:14<00:00, 15.43it/s]

[FINETUNE] train loss: 0.5968595363552569, val loss: 1.1508549515340132, accuracy: 0.6318418314255984
[FINETUNE] Epoch: 23

Finetune loss: 0.59255: 100%|██████████| 919/919 [02:20<00:00,  6.54it/s]
Finetune val loss: 1.24611: 100%|██████████| 226/226 [00:14<00:00, 15.49it/s]

[FINETUNE] train loss: 0.587764928765315, val loss: 1.2746178244618216, accuracy: 0.6245577523413112
[FINETUNE] Epoch: 24

Finetune loss: 0.58058: 100%|██████████| 919/919 [02:20<00:00,  6.56it/s]
Finetune val loss: 0.76107: 100%|██████████| 226/226 [00:14<00:00, 15.47it/s]

[FINETUNE] train loss: 0.5799075120742644, val loss: 1.172872437179564, accuracy: 0.6607700312174818
[FINETUNE] Epoch: 25

Finetune loss: 0.59785: 100%|██████████| 919/919 [02:20<00:00,  6.55it/s]
Finetune val loss: 0.86428: 100%|██████████| 226/226 [00:14<00:00, 15.42it/s]

[FINETUNE] train loss: 0.5756236260911987, val loss: 1.287433969829796, accuracy: 0.6230315643426986
[FINETUNE] Epoch: 26

Finetune loss: 0.59843: 100%|██████████| 919/919 [02:20<00:00,  6.55it/s]
Finetune val loss: 0.90232: 100%|██████████| 226/226 [00:14<00:00, 15.50it/s]

[FINETUNE] train loss: 0.5698320506770259, val loss: 1.1434333728412518, accuracy: 0.6616024973985432
[FINETUNE] Epoch: 27

Finetune loss: 0.58663: 100%|██████████| 919/919 [02:19<00:00,  6.57it/s]
Finetune val loss: 0.71581: 100%|██████████| 226/226 [00:14<00:00, 15.45it/s]

[FINETUNE] train loss: 0.5655228196114512, val loss: 1.187530158325239, accuracy: 0.6625043357613597
[FINETUNE] Epoch: 28

Finetune loss: 0.54717: 100%|██████████| 919/919 [02:19<00:00,  6.58it/s]
Finetune val loss: 0.74357: 100%|██████████| 226/226 [00:14<00:00, 15.65it/s]

[FINETUNE] train loss: 0.5627344370082986, val loss: 1.1763819356201177, accuracy: 0.6536246964967048
[FINETUNE] Epoch: 29

Finetune loss: 0.69592: 100%|██████████| 919/919 [02:19<00:00,  6.57it/s]
Finetune val loss: 0.74683: 100%|██████████| 226/226 [00:14<00:00, 15.62it/s]

[FINETUNE] train loss: 0.561272705840753, val loss: 1.2299307731138183, accuracy: 0.643565730142213
[EARLY STOPPING] Validation loss hasn't improved for 10 epochs.
x`