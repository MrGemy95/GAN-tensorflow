# Generative adversarial networks 

This's a reimplementation of [generative adversarial networks](https://arxiv.org/abs/1406.2661) using tensorflow.
## Usage
- download fashion-mnist in "data" folder

- run ``` python gan.py ```

### Training monitoring 

Discriminator loss            |  Generator loss
:-------------------------:|:-------------------------:
![](imgs/loss1.png?raw=true)  |  ![](imgs/loss2.png?raw=true  )


### Model Output samples
This's the network output during the training.

![](imgs/1.png?raw=true)  ![](imgs/3.png?raw=true  ) ![](imgs/2.png?raw=true  ) ![](imgs/4.png?raw=true  )


### References
- https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners
- used the same arcitecture and hyperparameters of [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections )

