# Writer Identifier

### A ML Model for Writer Identification.

Writer identification problem is the problem of assigning a known writer
to an unknown text script from a group of possible writers. This is
useful in many areas of interest such as forensics, security, historical
documents analysis and many more. Writer identification has seen a
plethora of research and attention from researchers in recent years,
however, it still remains a difficult problem to solve due to variations
in writing style and conditions, and the problems arising from
variations in image quality due to document degradation and other
incidental factors.

Proposed Method
===============

The pipeline, shown in the following figure, shows the main modules that the data goes through
before making a decision: Preprocessing, Feature Extraction and
Classification.

First, each image is processed to retrieve the handwritten paragraph
only to reduce the overall computational complexity and remove other
irrelevant information from the image so that only the handwritten part
affects the decision making process.

In feature extraction module, the implementation was based on the
methodology in [1] as their feature vector was based on Local Binary
Pattern (LBP) which is used in many computer vision problems due to its
robustness against noise. Authors in [1] introduced their own variation of LBP,
Sparse Radial Sampling Local Binary Pattern (SRS-LBP), which allows for
sampling of the circular patterns of traditional LBP at a lower
computational cost. In addition, the comparison operator of the
traditional LBP is changed to a threshold that is derived statistically
from each image. Then the feature vector is formed as the histogram of
each SRS-LBP at different radii, concatenated together after removing
the 0 pattern which corresponds to foreground and background only
patterns, thus providing no information. 

Then the features are normalized using L2 normalization before applying a Nearest Neighbour
classification to produce the resulting label.

![Pipeline](https://user-images.githubusercontent.com/32196766/105079726-7d617580-5a98-11eb-9ab9-1f582dcd6deb.png)

Experimental Results
====================
We analyzed the performance of our algorithm based on IAM dataset.  We worked with complete form text only. 

Each iteration, three different randomwriters are chosen from the set.  For each writer, two random pages are chosen for training and theothers are used for testing.

![Results Table](https://user-images.githubusercontent.com/32196766/105080743-edbcc680-5a99-11eb-8d2a-8d300c5196d2.png)


References
==========
[1] A. Nicolaou, A. D. Bagdanov, M. Liwicki, and D. Karatzas, “Sparse
radial sampling lbp for writer identification” ICDAR, 2015.