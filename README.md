# GAN-training-stabilization-techniques

# Description:
Training Generative Adversarial Networks (GANs) can be notoriously unstable, with issues such as mode collapse (where the generator produces the same output for all inputs) or gradient vanishing/exploding. Several training stabilization techniques have been proposed to mitigate these issues, including techniques like gradient penalty, label smoothing, and two-time scale updates. In this project, we will explore these techniques and implement them to stabilize GAN training.

We will demonstrate some stabilization techniques using a simple GAN model.

# ✅ What It Does:
* Gradient penalty is added to the discriminator's loss to stabilize the GAN training by penalizing the gradients that deviate from 1.

* The model is trained using a min-max loss function for both the discriminator and generator.

* Generates images and displays them periodically to visualize the progress of the GAN training.

# Key features:
* Gradient penalty helps to stabilize the training of GANs by ensuring smooth gradients for the discriminator.

* Adversarial loss is combined with gradient penalty to balance the training between the generator and discriminator.

* The model can generate fashion-like images after training, using the Fashion-MNIST dataset.
