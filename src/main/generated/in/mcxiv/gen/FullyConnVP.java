package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class FullyConnVP extends VP {
    public FullyConnVP(int num_neurons) {
        super("num_neurons", num_neurons);
        add("type", "fc");
    }

    public FullyConnVP num_neurons(int num_neurons) {
        add("num_neurons", num_neurons);
        return this;
    }

    public int num_neurons() {
        int num_neurons = getInt("num_neurons");
        return num_neurons;
    }

    public FullyConnVP l1_decay_mul(double l1_decay_mul) {
        add("l1_decay_mul", l1_decay_mul);
        return this;
    }

    public FullyConnVP l2_decay_mul(double l2_decay_mul) {
        add("l2_decay_mul", l2_decay_mul);
        return this;
    }

    public FullyConnVP bias_pref(double bias_pref) {
        add("bias_pref", bias_pref);
        return this;
    }

    public double l1_decay_mul() {
        double l1_decay_mul = getD("l1_decay_mul", 0.0);
        return l1_decay_mul;
    }

    public double l2_decay_mul() {
        double l2_decay_mul = getD("l2_decay_mul", 1.0);
        return l2_decay_mul;
    }

    public double bias_pref() {
        double bias_pref = getD("bias_pref", 0.0);
        return bias_pref;
    }

    public FullyConnVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
