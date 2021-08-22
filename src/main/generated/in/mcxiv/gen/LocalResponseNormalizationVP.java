package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class LocalResponseNormalizationVP extends VP {
    public LocalResponseNormalizationVP(int k, int n, double alpha, double beta) {
        super("k", k, "n", n, "alpha", alpha, "beta", beta);
        add("type", "lnr");
    }

    public LocalResponseNormalizationVP k(int k) {
        add("k", k);
        return this;
    }

    public LocalResponseNormalizationVP n(int n) {
        add("n", n);
        return this;
    }

    public LocalResponseNormalizationVP alpha(double alpha) {
        add("alpha", alpha);
        return this;
    }

    public LocalResponseNormalizationVP beta(double beta) {
        add("beta", beta);
        return this;
    }

    public int k() {
        int k = getInt("k");
        return k;
    }

    public int n() {
        int n = getInt("n");
        return n;
    }

    public double alpha() {
        double alpha = getD("alpha");
        return alpha;
    }

    public double beta() {
        double beta = getD("beta");
        return beta;
    }

    public LocalResponseNormalizationVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
