package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class TanhVP extends VP {
    public TanhVP() {
        super();
        add("type", "tanh");
    }

    public TanhVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
