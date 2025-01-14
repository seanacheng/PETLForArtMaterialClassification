Our chosen prediction task is image classification. More specifically, we have a dataset of individu-
ally photographed art pieces in the Rijksmuseum, with associated labeling, Mensink & Van Gemert
(2014). We want to build a model that can accurately predict what material each art piece is made
from. Our model considers 15 different materials (e.g. paper, silver, porcelain, etc.), and all of the
input data is drawn from these same classes. Our models take these images, and output predicted
probabilities for each of the 15 classes.

Our goal was to evaluate the effectiveness and generalizability of vision transformer models with
transfer learning. Using transfer learning allows pretraining our models on substantially different
datasets than the task we are attempting to solve, in the regime of very limited data for the target task.
The source task involved training on the popular Imagenet1k Deng et al. (2009) dataset, a dataset
of ∼ 1.3 million natural images from the "real world". This task involves classifying the input
images as 1 of 1000 classes (e.g. cars, dogs, guitars, etc.). In contrast, our target task dataset, the
Rijksmuseum challenge dataset Mensink & Van Gemert (2014), is comprised of images of artworks.
This project extends the work of Tonkes and Sabatelli Tonkes & Sabatelli (2022), which investigates
and evaluates transfer learning with multiple different vision transformer model (VTs) architectures
and convolutional neural network architectures. Our goal is to evaluate which of our methods and
architectures perform the best.