package in.mcxiv.ai.convnet;

import in.mcxiv.ai.convnet.net.VPL;
import in.mcxiv.ai.convnet.trainers.Trainer;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

@Disabled
public class TestOnIris {

    @Test
    void well_here_we_go() {

        Dataset dataset = Dataset.IRIS;
        dataset.load();
        Vol[] xi = dataset.xi;
        int[] yi = dataset.yi;

        VPL layer_defs = new VPL();
        layer_defs.push("type", "input", "out_sx", xi[0].sx, "out_sy", 1, "out_depth", 1);
        layer_defs.push("type", "conv", "sx", 5, "filters", 8, "stride", 1, "pad", 2, "activation", "relu");
        layer_defs.push("type", "pool", "sx", 2, "stride", 2);
        layer_defs.push("type", "conv", "sx", 5, "filters", 16, "stride", 1, "pad", 2, "activation", "relu");
        layer_defs.push("type", "pool", "sx", 3, "stride", 3);
        layer_defs.push("type", "softmax", "num_classes", 3);
        Net net = new Net();
        net.makeLayers(layer_defs);

        Util.printARandomExample(xi, yi, net);

//        var trainer = new Trainer(net, "learning_rate", 1000, "method", "adadelta", "batch_size", 20, "l2_decay", 0.001);
        Trainer trainer = new Trainer(net, "learning_rate", 0.1, "method", "sgd", "batch_size", 20, "l2_decay", 0.001);
        for (int i = 0, s = 1000, r = s / 10; i < 1000; i++) {
            int tidx = i % xi.length;
            Vol x = xi[tidx];
            int y = yi[tidx];
            System.out.println("cost_loss = " + trainer.train(x, y).getD("cost_loss"));
            if (i % r == 0)
                Util.printARandomExample(xi, yi, net);

        }

        System.out.println("<<<<<<<<<<<<< >>>>>>>>>>>>>");
        for (int i = 0; i < 10; i++)
            Util.printARandomExample(xi, yi, net);

    }
}
