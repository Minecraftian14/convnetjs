package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class TrainerVP extends VP {
    public TrainerVP() {
        super();
        add("type", "trainer");
    }

    public TrainerVP learning_rate(double learning_rate) {
        add("learning_rate", learning_rate);
        return this;
    }

    public TrainerVP l1_decay(double l1_decay) {
        add("l1_decay", l1_decay);
        return this;
    }

    public TrainerVP l2_decay(double l2_decay) {
        add("l2_decay", l2_decay);
        return this;
    }

    public TrainerVP batch_size(int batch_size) {
        add("batch_size", batch_size);
        return this;
    }

    public TrainerVP method(String method) {
        add("method", method);
        return this;
    }

    public TrainerVP momentum(double momentum) {
        add("momentum", momentum);
        return this;
    }

    public TrainerVP ro(double ro) {
        add("ro", ro);
        return this;
    }

    public TrainerVP eps(double eps) {
        add("eps", eps);
        return this;
    }

    public TrainerVP beta1(double beta1) {
        add("beta1", beta1);
        return this;
    }

    public TrainerVP beta2(double beta2) {
        add("beta2", beta2);
        return this;
    }

    public double learning_rate() {
        double learning_rate = getD("learning_rate", 0.01);
        return learning_rate;
    }

    public double l1_decay() {
        double l1_decay = getD("l1_decay", 0.0);
        return l1_decay;
    }

    public double l2_decay() {
        double l2_decay = getD("l2_decay", 0.0);
        return l2_decay;
    }

    public int batch_size() {
        int batch_size = getInt("batch_size", 1);
        return batch_size;
    }

    public String method() {
        String method = getSt("method", "sgd");
        return method;
    }

    public double momentum() {
        double momentum = getD("momentum", 0.9);
        return momentum;
    }

    public double ro() {
        double ro = getD("ro", 0.95);
        return ro;
    }

    public double eps() {
        double eps = getD("eps", 1e-8);
        return eps;
    }

    public double beta1() {
        double beta1 = getD("beta1", 0.9);
        return beta1;
    }

    public double beta2() {
        double beta2 = getD("beta2", 0.999);
        return beta2;
    }

    public TrainerVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
