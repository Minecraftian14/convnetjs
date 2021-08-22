package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class DropoutVP extends VP {
    public DropoutVP() {
        super();
        add("type", "dropout");
    }

    public DropoutVP drop_prob(double drop_prob) {
        add("drop_prob", drop_prob);
        return this;
    }

    public double drop_prob() {
        double drop_prob = getD("drop_prob", 0.5);
        return drop_prob;
    }

    public DropoutVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
