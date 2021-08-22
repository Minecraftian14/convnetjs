package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class SigmoidVP extends VP {
    public SigmoidVP() {
        super();
        add("type", "sigmoid");
    }

    public SigmoidVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
