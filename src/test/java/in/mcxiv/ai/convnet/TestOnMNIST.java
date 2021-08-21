package in.mcxiv.ai.convnet;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.ai.convnet.trainers.Trainer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public class TestOnMNIST {

    public static final int trainingSamples = 10000;
    public static Vol[] xi = new Vol[trainingSamples];
    public static int[] yi = new int[trainingSamples];

    @BeforeEach
    public void datasetInitialization() {
        try {

            BufferedReader reader = Files.newBufferedReader(Paths.get(TestOnMNIST.class.getClassLoader().getResource("mnist_train.csv").toURI()));
            CSVReader csvReader = new CSVReader(reader);

            String[] line;

            csvReader.readNext();//skip the first row

            for (int i = 0; i < trainingSamples && (line = csvReader.readNext()) != null; i++) {
                Vol x = new Vol(28, 28, 1);
                for (int xidx = 0; xidx < 784; xidx++) {
                    int a = xidx % 28;
                    int b = xidx / 28;
                    x.set(a, b, 0, Double.parseDouble(line[xidx]) / 255);
                }
                xi[i] = x;
                yi[i] = Integer.parseInt(line[784]);
            }

            reader.close();
            csvReader.close();

        } catch (IOException | CsvValidationException | URISyntaxException e) {
            e.printStackTrace();
        }
    }

    public static class AutoEncoderTest {

        public static void main(String[] args) {
            new TestOnMNIST().datasetInitialization();

            var layer_defs = new VP.VPL();
            layer_defs.push("type", "input", "out_sx", 28, "out_sy", 28, "out_depth", 1);
            layer_defs.push("type", "fc", "num_neurons", 50, "activation", "tanh");
            layer_defs.push("type", "fc", "num_neurons", 50, "activation", "tanh");
            layer_defs.push("type", "fc", "num_neurons", 2);
            layer_defs.push("type", "fc", "num_neurons", 50, "activation", "tanh");
            layer_defs.push("type", "fc", "num_neurons", 50, "activation", "tanh");
            layer_defs.push("type", "regression", "num_neurons", 28 * 28);

            Net net = new Net();
            net.makeLayers(layer_defs);

            var trainer = new Trainer(net, "learning_rate", 1, "method", "adadelta", "batch_size", 50, "l2_decay", 0.001, "l1_decay", 0.001);

            Supplier<BufferedImage> getOriginal = () -> {
                int idx = (int) (xi.length * Math.random());
                System.out.println("Expected image " + yi[idx]);
                return VolUtil.vol_1_to_img(net.forward(xi[idx]));
            };
            Runnable runnable = ImageDisplayUtility.display("Test Autoencoder", getOriginal, 0, 0);

            while (true) {
                for (var i = 0; i < xi.length; i++) {
                    var x = xi[i];
//                    System.out.println("trainer.train(x, x.w).get(\"loss\") = " + trainer.train(x, x.w).get("loss"));
                    trainer.train(x, x.w);
                }
                runnable.run();
            }
        }
    }

    @Test
    void classifyTest() {

        var layer_defs = new VP.VPL();
        layer_defs.push("type", "input", "out_sx", 28, "out_sy", 28, "out_depth", 1);
        layer_defs.push("type", "conv", "sx", 5, "filters", 8, "stride", 1, "pad", 2, "activation", "relu");
        layer_defs.push("type", "pool", "sx", 2, "stride", 2);
        layer_defs.push("type", "conv", "sx", 5, "filters", 16, "stride", 1, "pad", 2, "activation", "relu");
        layer_defs.push("type", "pool", "sx", 3, "stride", 3);
        layer_defs.push("type", "softmax", "num_classes", 10);
        Net net = new Net();
        net.makeLayers(layer_defs);

        var trainer = new Trainer(net, "method", "adadelta", "batch_size", 20, "l2_decay", 0.001);

        Util.printARandomExample(xi, yi, net);

        for (int i = 0; i < 10000; i++) {
            int tidx = i % trainingSamples;
            var x = xi[tidx];
            var y = yi[tidx];
            trainer.train(x, y);
        }

        System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        for (int i = 0; i < 10; i++) {
            Util.printARandomExample(xi, yi, net);
        }
        System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

        double met = 0;
        for (int j = 0; j < xi.length; j++) {
            Vol x = xi[j];
            Vol pred = net.forward(x);
            double max = -Double.MAX_VALUE;
            int maxi = -1;
            for (int i = 0; i < pred.depth; i++) {
                double v = pred.get(0, 0, i);
                if (v >= max) {
                    max = v;
                    maxi = i;
                }
            }
            if (maxi == yi[j]) {
                met++;
            }
        }
        System.out.println("met = " + met);
        System.out.println("L   = " + (met / xi.length));

    }


}
