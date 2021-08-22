package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class RegressionVP extends VP {
    public RegressionVP(int num_neurons) {
        super("num_neurons", num_neurons);
        add("type", "regression");
    }

    public RegressionVP num_neurons(int num_neurons) {
        add("num_neurons", num_neurons);
        return this;
    }

    public int num_neurons() {
        int num_neurons = getInt("num_neurons");
        return num_neurons;
    }

    public RegressionVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
