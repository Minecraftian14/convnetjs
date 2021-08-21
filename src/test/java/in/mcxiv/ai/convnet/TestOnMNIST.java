package in.mcxiv.ai.convnet;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.ai.convnet.trainers.Trainer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class TestOnMNIST {

    public static final int trainingSamples = 1000;
    public Vol[] xi = new Vol[trainingSamples];
    public int[] yi = new int[trainingSamples];

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
                    x.set(a, b, 0, Double.parseDouble(line[xidx]));
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

    Net net;

    @BeforeEach
    void networkInitialization() {
        var layer_defs = new VP.VPL();
        layer_defs.push("type", "input", "out_sx", 28, "out_sy", 28, "out_depth", 1);
        layer_defs.push("type", "conv", "sx", 5, "filters", 8, "stride", 1, "pad", 2, "activation", "relu");
        layer_defs.push("type", "pool", "sx", 2, "stride", 2);
        layer_defs.push("type", "conv", "sx", 5, "filters", 16, "stride", 1, "pad", 2, "activation", "relu");
        layer_defs.push("type", "pool", "sx", 3, "stride", 3);
        layer_defs.push("type", "softmax", "num_classes", 10);
        net = new Net();
        net.makeLayers(layer_defs);
    }

    @Test
    void mainTest() {

        var trainer = new Trainer(net, "method", "adadelta", "batch_size", 20, "l2_decay", 0.001);

        Util.printARandomExample(xi, yi, net);

        for (int i = 0; i < 1000; i++) {
            int tidx = i % trainingSamples;
            var x = xi[tidx];
            var y = yi[tidx];
            trainer.train(x, y);
        }

        for (int i = 0; i < 10; i++) {

            Util.printARandomExample(xi, yi, net);
        }

    }

}
