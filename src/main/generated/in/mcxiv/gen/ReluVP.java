package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class ReluVP extends VP {
    public ReluVP() {
        super();
        add("type", "relu");
    }

    public ReluVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
