# Deep Learning Do It Yourself!

\blurb{
    This site collects resources to learn Deep Learning in the form of
    Modules available through the sidebar on the left.
    As a student, you can walk through the modules at your own pace and
    interact with others thanks to the associated [Discord server](https://discord.gg/nZQ3fe3). You don’t need any special hardware or software.
}

## Practical deep learning course

The main goal of the course is to allow students to understand papers, blog posts and codes available online and to adapt them to their projects as soon as possible. In particular, we avoid the use of any high-level neural networks API and focus on the [PyTorch](https://pytorch.org/) library in Python.

The course is divided into sessions (containing possibly several modules), each session requiring a significant amount of coding. At the end of this course, students were able to read very recent papers and reproduce (or even ameliorate) their experiments. 

All the code used in this course is available on the GitHub repository [dataflowr/notebooks](https://github.com/dataflowr/notebooks). You will find the solutions to the practicals on this repo! You can fork the repo if you want to run the code locally: [GitHub Docs about fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) then follow the steps in [Module 0](./modules/0-sotfware-installation/). Most of the code will not require a GPU. 

:warning: When a GPU is required , you can launch the code on colab by following the corresponding link given in the module (see for example [Module 1](./modules/1-intro-general-overview/)).

Pre-requisites:

- Mathematics: basics of linear algebra, probability, differential calculus and optimization
- Programming: Python. Test your proficiency: [quiz](https://dataflowr.github.io/quiz/python.html)

### :sunflower: Session 1 - Finetuning VGG

Start right away and train a deep neural network on a GPU with [Module 1 - Introduction & General Overview](./modules/1-intro-general-overview/)

Be sure to build your own classifier with more dogs and cats in the practicals.
~~~
<details>
  <summary>Things to remember</summary>
~~~
> - you do not need to understand everything to run a deep learning model! But the main goal of this course will be to come back to each step done today and understand them...
> - to use the dataloader from Pytorch, you need to follow the API (i.e. for classification store your dataset in folders)
> - using a pretrained model and modifying it to adapt it to a similar task is easy. 
> - if you do not understand why we take this loss, that's fine, we'll cover that in Module 3.
> - even with a GPU, avoid unnecessary computations!
~~~
</details>
~~~

### :sunflower: Session 2 - PyTorch tensors and Autodiff

- [Module 2a - PyTorch tensors](https://dataflowr.github.io/website/modules/2a-pytorch-tensors/)
- [Module 2b - Automatic differentiation](https://dataflowr.github.io/website/modules/2b-automatic-differentiation/) + Practicals
- MLP from scratch start of [HW1](https://dataflowr.github.io/website/homework/1-mlp-from-scratch/) 
- [another look at autodiff with dual numbers and Julia](https://github.com/dataflowr/notebooks/blob/master/Module2/AD_with_dual_numbers_Julia.ipynb)
~~~
<details>
  <summary>Things to remember</summary>
~~~
>- Pytorch tensors = Numpy on GPU + gradients!
>- in deep learning, [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) is used everywhere. The rules are the same as for Numpy.
>- Automatic differentiation is not only the chain rule! Backpropagation algorithm (or dual numbers) is a clever algorithm to implement automatic differentiation...
~~~
 </details>
~~~
### :sunflower: Session 3
- [Module 3 - Loss function for classification](https://dataflowr.github.io/website/modules/3-loss-functions-for-classification/) 
- [Module 4 - Optimization for deep learning](https://dataflowr.github.io/website/modules/4-optimization-for-deep-learning/)
- [Module 5 - Stacking layers](https://dataflowr.github.io/website/modules/5-stacking-layers/) and overfitting a MLP on CIFAR10
- [Module 6: Convolutional neural network](https://dataflowr.github.io/website/modules/6-convolutional-neural-network/)
- how to regularize with dropout and uncertainty estimation with MC Dropout: [Module 15 - Dropout](https://dataflowr.github.io/website/modules/15-dropout/)
~~~
<details>
  <summary>Things to remember</summary>
~~~
>- Loss vs Accuracy. Know your loss for a classification task!
>- know your optimizer (Module 4)
>- know how to build a neural net with torch.nn.module (Module 5)
>- know how to use convolution and pooling layers (kernel, stride, padding)
>- know how to use dropout 
~~~
</details>
~~~
### :sunflower: Session 4
- [Module 7 - Dataloading](https://dataflowr.github.io/website/modules/7-dataloading/)
- [Module 8a - Embedding layers](https://dataflowr.github.io/website/modules/8a-embedding-layers/)
- [Module 8b - Collaborative filtering](https://dataflowr.github.io/website/modules/8b-collaborative-filtering/) and build your own recommender system: [08\_collaborative\_filtering\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_empty.ipynb) (on a larger dataset [08\_collaborative\_filtering\_1M.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_1M.ipynb))
- [Module 8c - Word2vec](https://dataflowr.github.io/website/modules/8c-word2vec/) and build your own word embedding [08\_Word2vec\_pytorch\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_Word2vec_pytorch_empty.ipynb)
- [Module 16 - Batchnorm](https://dataflowr.github.io/website/modules/16-batchnorm/) and check your understanding with [16\_simple\_batchnorm\_eval.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_simple_batchnorm_eval.ipynb) and more [16\_batchnorm\_simple.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_batchnorm_simple.ipynb)
- [Module 17 - Resnets](https://dataflowr.github.io/website/modules/17-resnets/)
- start of [Homework 2: Class Activation Map and adversarial examples](https://dataflowr.github.io/website/homework/2-CAM-adversarial/)
~~~
<details>
  <summary>Things to remember</summary>
~~~
> - know how to use dataloader
> - to deal with categorical variables in deep learning, use embeddings
> - in the case of word embedding, starting in an unsupervised setting, we built a supervised task (i.e. predicting central / context words in a window) and learned the representation thanks to negative sampling
> - know your batchnorm
> - architectures with skip connections allows deeper models
~~~
</details>
~~~
### :sunflower: Session 5
- [Module 9a: Autoencoders](https://dataflowr.github.io/website/modules/9a-autoencoders/) and code your noisy autoencoder [09\_AE\_NoisyAE.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/09_AE_NoisyAE.ipynb)
- [Module 10: Generative Adversarial Networks]() and code your GAN, Conditional GAN and InfoGAN [10\_GAN\_double\_moon.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module10/10_GAN_double_moon.ipynb)
- [Module 13: Siamese Networks and Representation Learning](https://dataflowr.github.io/website/modules/13-siamese/)
- start of [Homework 3: VAE for MNIST clustering and generation](https://dataflowr.github.io/website/homework/3-VAE/)
### :sunflower: Session 6
- [Module 11a - Recurrent Neural Networks theory](https://dataflowr.github.io/website/modules/11a-recurrent-neural-networks-theory/)
- [Module 11b - Recurrent Neural Networks practice](https://dataflowr.github.io/website/modules/11b-recurrent-neural-networks-practice/) and predict engine failure with [11\_predicitions\_RNN\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module11/11_predicitions_RNN_empty.ipynb)
- [Module 11c - Batches with sequences in Pytorch](https://dataflowr.github.io/website/modules/11c-batches-with-sequences/)

### :sunflower: Session 7
- [Module 12 - Attention and Transformers](https://dataflowr.github.io/website/modules/12-attention/)
- Correcting the PyTorch tutorial on attention in seq2seq: [12\_seq2seq\_attention.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module12/12_seq2seq_attention.ipynb)
- Build your own microGPT: [GPT\_hist.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module12/GPT_hist.ipynb)

### :sunflower: Session 8
- [Module 9b - UNets](https://dataflowr.github.io/website/modules/9b-unet/)
- [Module 9c - Flows](https://dataflowr.github.io/website/modules/9c-flows/)
- Build your own Real NVP: [Normalizing\_flows\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/Normalizing_flows_empty.ipynb)
### :sunflower: Session 9
- []Module 18a - Denoising Score Matching for Energy Based Models](https://dataflowr.github.io/website/modules/18a-energy/)
- [Module 18b - Denoising Diffusion Probabilistic Models](https://dataflowr.github.io/website/modules/18b-diffusion/)
- Train your own DDPM on MNIST: [ddpm\_nano\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_empty.ipynb)
- Finetuning on CIFAR10: [ddpm\_micro\_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_micro_sol.ipynb)

For more updates: [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/marc_lelarge.svg?style=social&label=Follow%20%40marc_lelarge)](https://twitter.com/marc_lelarge) 

and check the 
# [GitHub repository: dataflowr/notebooks](https://github.com/dataflowr/notebooks)
## Curators

[Marc Lelarge](https://www.di.ens.fr/~lelarge/),  [Andrei Bursuc](https://abursuc.github.io/) with [Jill-Jênn Vie](https://jill-jenn.net/)

## Course in a hurry


**Super fast track** to learn the basics of deep learning from scratch:
- Have a look at the [slides](https://dataflowr.github.io/slides/module1.html) of [Module 1: Introduction & General Overview](./modules/1-intro-general-overview)
- Run the [notebook](https://github.com/dataflowr/notebooks/blob/master/Module2/02a_basics.ipynb) (or in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module2/02a_basics.ipynb)) of [Module 2a: Pytorch Tensors](./modules/2a-pytorch-tensors)
- Run the [notebook](https://github.com/dataflowr/notebooks/blob/master/Module2/02b_linear_reg.ipynb) (or in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module2/02b_linear_reg.ipynb)) of [Module 2b: Automatic Differentiation](./modules/2b-automatic-differentiation)
- Check the [Minimal working examples](./modules/3-loss-functions-for-classification/#minimal_working_examples) of [Module 3: Loss functions for classification](./modules/3-loss-functions-for-classification). If you do not understand, have a look at the [slides](https://dataflowr.github.io/slides/module3.html).
- Have a look at the [slides](https://dataflowr.github.io/slides/module4.html) of [Module 4: Optimization for Deep Learning](./modules/4-optimization-for-deep-learning)
- Try playback speed 1.5 for the [video](https://youtu.be/OiyZXdnLHcI?t=149) from [Module 5: Stacking layers](./modules/5-stacking-layers).
- Run the [notebook](https://github.com/dataflowr/notebooks/blob/master/Module6/06_convolution_digit_recognizer.ipynb) (or in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module6/06_convolution_digit_recognizer.ipynb)) of [Module 6: Convolutional Neural Network](./modules/6-convolutional-neural-network)
- Try playback speed 2 for the [video](https://youtu.be/vm-ZusIUkiY?t=133) from [Module 7: Dataloading](./modules/7-dataloading)
- Have a look at the [slides](https://dataflowr.github.io/slides/module8a.html) of [Module 8a: Embedding layers](./modules/8a-embedding-layers)
- Well done! Now you have time to enjoy deep learning!

<!-- ### Annotation tool

- [hypothes.is](https://hypothes.is/groups/EzzjE8gb/deep-learning-ens-2020) allows you to annotate this website and the web in general. You'll find some hints for the practicals here!
-->


## For contributors

Join the [GitHub repo dataflowr](https://github.com/dataflowr) and make a pull request. [What are pull requests?](https://yangsu.github.io/pull-request-tutorial/)

Thanks to [Daniel Huynh](https://github.com/dhuynh95), [Eric Daoud](https://github.com/ericdaat), [Simon Coste](https://github.com/SimonCoste)
<!-- to be updated
## Modules

- [Module 0: Software installation](./modules/0-sotfware-installation)
- [Module 1: Introduction & General Overview](./modules/1-intro-general-overview)
- [Module 2a: Pytorch Tensors](./modules/2a-pytorch-tensors)
- [Module 2b: Automatic Differentiation](./modules/2b-automatic-differentiation)
- [Module 3: Loss functions for classification](./modules/3-loss-functions-for-classification)
- [Module 4: Optimization for Deep Learning](./modules/4-optimization-for-deep-learning)
- [Module 5: Stacking layers](./modules/5-stacking-layers)
- [Module 6: Convolutional Neural Network](./modules/6-convolutional-neural-network)
- [Module 7a: Embedding layers and dataloaders](./modules/7a-embedding-layers-dataloaders)
- [Module 7b: Collaborative Filtering](./modules/7b-collaborative-filtering)
- [Modules 8: Autoencoders](./modules/8-autoencoders)
- [Module 9: Generative Adversarial Networks](./modules/9-generative-adversarial-networks)
- [Module 10a: Recurrent Neural Networks theory](./modules/10a-recurrent-neural-networks-theory)
- [Module 10b: Recurrent Neural Networks practice](./modules/10b-recurrent-neural-networks-practice)
-->


Materials from this site is used for courses at ENS and X. 

~~~
<img src="/assets/ENS_logo.jpg" style="width: 400px; height: auto; display: inline">
<img src="/assets/X_logo.png" style="margin-left:1em; width: 180px; height: auto; display: inline">
~~~
