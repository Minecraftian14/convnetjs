package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class InputVP extends VP {
    public InputVP(int out_depth) {
        super("out_depth", out_depth);
        add("type", "input");
    }

    public InputVP out_depth(int out_depth) {
        add("out_depth", out_depth);
        return this;
    }

    public int out_depth() {
        int out_depth = getInt("out_depth");
        return out_depth;
    }

    public InputVP out_sx(int out_sx) {
        add("out_sx", out_sx);
        return this;
    }

    public InputVP out_sy(int out_sy) {
        add("out_sy", out_sy);
        return this;
    }

    public int out_sx() {
        int out_sx = getInt("out_sx", 1);
        return out_sx;
    }

    public int out_sy() {
        int out_sy = getInt("out_sy", 1);
        return out_sy;
    }

    public InputVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
