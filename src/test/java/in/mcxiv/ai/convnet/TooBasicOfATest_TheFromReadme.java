package in.mcxiv.ai.convnet;

import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.ai.convnet.trainers.Trainer;
import org.junit.jupiter.api.Test;

class TooBasicOfATest_TheFromReadme {

    @Test
    void Two_layer_neural_network() {

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

    }

    @Test
    void Convolutional_Neural_Network() {

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
        var x = VolUtil.img_to_vol(null, true);
        var output_probabilities_vol = net.forward(x);

    }

    // https,//cs.stanford.edu/people/karpathy/convnetjs/started.html
    @Test
    void Getting_Started_Neural_Net_Classification() {
        var layer_defs = new VP.VPL();
        // input layer of size 1x1x2 (all volumes are 3D)
        layer_defs.push("type", "input", "out_sx", 1, "out_sy", 1, "out_depth", 2);
        // some fully connected layers
        layer_defs.push("type", "fc", "num_neurons", 20, "activation", "relu");
        layer_defs.push("type", "fc", "num_neurons", 20, "activation", "relu");
        // a softmax classifier predicting probabilities for two classes, 0,1
        layer_defs.push("type", "softmax", "num_classes", 2);

        // create a net out of it
        var net = new Net();
        net.makeLayers(layer_defs);

        // the network always works on Vol() elements. These are essentially
        // simple wrappers around lists, but also contain gradients and dimensions
        // line below will create a 1x1x2 volume and fill it with 0.5 and -1.3
        var x = new Vol(0.5, -1.3);

        var probability_volume = net.forward(x);
        System.out.println("probability that x is class 0, " + probability_volume.w.get(0));
        // prints 0.50101

        var trainer = new Trainer(net, "learning_rate", 0.01, "l2_decay", 0.001);
        trainer.train(x, 0);

        var probability_volume2 = net.forward(x);
        System.out.println("probability that x is class 0, " + probability_volume2.w.get(0));
        // prints 0.50374
    }

    @Test
    void Getting_Started_Neural_Net_Regression() {

        var layer_defs = new VP.VPL();
        layer_defs.push("type","input", "out_sx",1, "out_sy",1, "out_depth",2);
        layer_defs.push("type","fc", "num_neurons",5, "activation","sigmoid");
        layer_defs.push("type","regression", "num_neurons",1);
        var net = new Net();
        net.makeLayers(layer_defs);

        var x = new Vol(0.5, -1.3);

        // train on this datapoint, saying [0.5, -1.3] should map to value 0.7,
        // note that in this case we are passing it a list, because in general
        // we may want to regress multiple outputs and in this special case we
        // used num_neurons,1 for the regression to only regress one.
        var trainer = new Trainer(net,
        "learning_rate",0.01, "momentum",0.0, "batch_size",1, "l2_decay",0.001);
        DoubleBuffer y = new DoubleBuffer();
        y.add(0.7);
        trainer.train(x, y);

        // evaluate on a datapoint. We will get a 1x1x1 Vol back, so we get the
        // actual output by looking into its 'w' field,
        var predicted_values = net.forward(x);
        System.out.println("predicted value, " + predicted_values.w.get(0));
    }
}





















