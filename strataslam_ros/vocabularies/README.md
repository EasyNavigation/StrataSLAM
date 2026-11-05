# Visual vocabularies

`tum_rgbd.bin` is the vocabulary that a profile names with `fusion.vocabulary: default` (ADR-0054): 9964 words,
trained with `strataslam vocab train` on ORB features of 892 colour images of the TUM RGB-D benchmark (every fifth
image of `freiburg1_xyz` and `freiburg2_xyz`, indoor offices). `scripts/train_default_vocabulary.sh` rebuilds it, byte
for byte (sha256 `212e9c4d99c15c08bb4df5fa0fa5d7a0622fcde7263a383bf6597be87c09e204`).

It is derived from the TUM RGB-D images, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): J. Sturm, N. Engelhard, F. Endres, W. Burgard and
D. Cremers, "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012,
<https://cvg.cit.tum.de/data/datasets/rgbd-dataset>. Changes: the images are not included; the file holds the
cluster centres of their ORB descriptors and the inverse document frequency of each word.

For environments unlike offices (outdoors, warehouses, very repetitive corridors), a vocabulary trained on images of
your own environment can recognise places better: `strataslam vocab train <image_folder> --out vocab.bin`, then
`fusion.vocabulary: vocab.bin` in the profile.
