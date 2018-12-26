### Colorization

This GitHub contains codes for Sampling Using Neural Network - Colorization. I made each python files self-executable if you have CUDA and PyTorch.

### Abstract

The main idea of this paper is to explore the possibilities of generating samples from the neural networks, mostly focusing on the colorization of the grey-scale images. I will compare the existing methods for colorization and explore the possibilities of using new generative modeling to the task of colorization. The contributions of this paper are to compare the existing structures with similar generating structures(Decoders) and to apply the novel structures including Conditional VAE(CVAE), Conditional Wasserstein GAN with Gradient Penalty(CWGAN-GP), CWGAN-GP with L1 reconstruction loss, Adversarial Generative Encoders(AGE) and Introspective VAE(IVAE). I trained these models using CIFAR-10 images. To measure the performance, I use Inception Score(IS) which measures how distinctive each image is and how diverse overall samples are as well as human eyes for CIFAR-10 images. It turns out that CVAE with L1 reconstruction loss and IVAE achieve the highest score in IS. CWGAN-GP with L1 tends to learn faster than CWGAN-GP, but IS does not increase from CWGAN-GP. CWGAN-GP tends to generate more diverse images than other models using reconstruction loss. Also, I figured out that the proper regularization plays a vital role in generative modeling.


### Author

Wonbong Jang (Wayne), This is based on my dissertation at MSc in Computational Statistics and Machine Learning at University College London. He is currently a visiting fellow at London School of Economics working with Prof. Milan Vojnovic. He is originally from South Korea, and he is quite interested in Neural Networks itself and its applications to various domains. Feel free to contact me if you have any questions about this paper (or machine learning in overall) - won (dot)jang1108(at)gmail (dot)com

