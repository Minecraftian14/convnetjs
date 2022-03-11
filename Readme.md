
# ConvNet~~JS~~Java

ConvNetJS is a Javascript implementation of Neural networks, together with nice browser-based demos.
Do check it out [ConvNetJS](https://github.com/karpathy/convnetjs)

While, this fork is, well, a java implementation I transliterated while learning Neural Networks 😁.

[![](https://img.shields.io/discord/872811194170347520?color=%237289da&logoColor=%23424549)](https://discord.gg/hZnHFGvU6W)

It's _supposed_ to be supporting:

- Common **Neural Network modules** (fully connected layers, non-linearities)
- Classification (SVM/Softmax) and Regression (L2) **cost functions**
- Ability to specify and train **Convolutional Networks** that process images
- An experimental **Reinforcement Learning** module, based on Deep Q Learning

For much more information, um, I have no determination on maintaining this project (and it seems, the original fork is also not maintained) feel free to contact me at Discord.
Or consider the main page at [convnetjs.com](http://convnetjs.com)

**Note**: I am not actively maintaining ConvNetJS anymore because I simply don't have time. I think the npm repo might not work at this point.

## ~~Online Demos~~
### I wish I could write some swing demos...
- [Convolutional Neural Network on MNIST digits](http://cs.stanford.edu/~karpathy/convnetjs/demo/mnist.html)
- [Convolutional Neural Network on CIFAR-10](http://cs.stanford.edu/~karpathy/convnetjs/demo/cifar10.html)
- [Toy 2D data](http://cs.stanford.edu/~karpathy/convnetjs/demo/classify2d.html)
- [Toy 1D regression](http://cs.stanford.edu/~karpathy/convnetjs/demo/regression.html)
- [Training an Autoencoder on MNIST digits](http://cs.stanford.edu/~karpathy/convnetjs/demo/autoencoder.html)
- [Deep Q Learning Reinforcement Learning demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
- [Image Regression ("Painting")](http://cs.stanford.edu/~karpathy/convnetjs/demo/image_regression.html)
- [Comparison of SGD/Adagrad/Adadelta on MNIST](http://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html)

## Example Code

Here's a minimum example of defining a **2-layer neural network** and training
it on a single data point:

```javascript
    // species a 2-layer neural network with one hidden layer of 20 neurons
    var layer_defs = new VP.VPL();
    // input layer declares size of input. here, 2-D data
    // ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
    // then the first two dimensions (sx, sy) will always be kept at size 1
    layer_defs.push("type", "input", "out_sx", 1, "out_sy", 1, "out_depth", 2);
    // declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
    layer_defs.push("type", "fc", "num_neurons", 20, "activation", "relu");
    // declare the linear classifier on top of the previous hidden layer
    layer_defs.push("type", "softmax", "num_classes", 10);

    var net = new Net();
    net.makeLayers(layer_defs);

    // forward a random data point through the network
    var x = new Vol(0.3, -0.5);
    var prob = net.forward(x);

    // prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
    System.out.println("probability that x is class 0, " + prob.w.get(0)); // prints 0.50101

    var trainer = new Trainer(net, "learning_rate", 0.01, "l2_decay", 0.001);
    for (int i = 0; i < 100; i++) {
        trainer.train(x, 0); // train the network, specifying that x is class zero
    }

    var prob2 = net.forward(x);
    System.out.println("probability that x is class 0, " + prob2.w.get(0));
    // now prints 0.50374, slightly higher than previous 0.50101, the networks
    // weights have been adjusted by the Trainer to give a higher probability to
    // the class we trained the network with (zero)
```

and here is a small **Convolutional Neural Network** if you wish to predict on images:

```javascript
    var layer_defs = new VP.VPL();
    layer_defs.push("type", "input", "out_sx", 32, "out_sy", 32, "out_depth", 3); // declare size of input
    // output Vol is of size 32x32x3 here
    layer_defs.push("type", "conv", "sx", 5, "filters", 16, "stride", 1, "pad", 2, "activation", "relu");
    // the layer will perform convolution with 16 kernels, each of size 5x5.
    // the input will be padded with 2 pixels on all sides to make the output Vol of the same size
    // output Vol will thus be 32x32x16 at this point
    layer_defs.push("type", "pool", "sx", 2, "stride", 2);
    // output Vol is of size 16x16x16 here
    layer_defs.push("type", "conv", "sx", 5, "filters", 20, "stride", 1, "pad", 2, "activation", "relu");
    // output Vol is of size 16x16x20 here
    layer_defs.push("type", "pool", "sx", 2, "stride", 2);
    // output Vol is of size 8x8x20 here
    layer_defs.push("type", "conv", "sx", 5, "filters", 20, "stride", 1, "pad", 2, "activation", "relu");
    // output Vol is of size 8x8x20 here
    layer_defs.push("type", "pool", "sx", 2, "stride", 2);
    // output Vol is of size 4x4x20 here
    layer_defs.push("type", "softmax", "num_classes", 10);
    // output Vol is of size 1x1x10 here

    var net = new Net();
    net.makeLayers(layer_defs);

    // helpful utility for converting images into Vols is included
    Image image = ImageIO.read(***);
    var x = VolUtil.img_to_vol(image, true);
    var output_probabilities_vol = net.forward(x);
```

## ~~Getting Started~~
Hey! Dont use this project consider using ConvnetJS instead! Neither is it maintained, nor anyone can help.
A [Getting Started](http://cs.stanford.edu/people/karpathy/convnetjs/started.html) tutorial is available on main page.

The full [Documentation](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html) can also be found there.


Must have:
```groovy
allprojects {
    repositories {
        maven { url 'https://jitpack.io' }
    }
}

```

To get the library only:
```groovy
testImplementation 'com.github.Minecraftian14:convnetjs:0.1.0'

```
To get the library and a few utilities:

```groovy
testImplementation 'com.github.Minecraftian14:convnetjs:0.1.0:featured'

```
These utilities contains a few extra classes to make things slightly easier to use.
For instance, the same example from above can be written as:
```js
LayerVPL layer_defs = new LayerVPL();
layer_defs.input(2);
layer_defs.fc(20).activation("relu");
layer_defs.softmax(10);

Net net = new Net();
net.makeLayers(layer_defs);

Trainer trainer = new Trainer(net, new TrainerVP().method("sgd").learning_rate(0.01).l2_decay(0.001));
```

## License
MIT
