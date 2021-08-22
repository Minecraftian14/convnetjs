package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class PoolVP extends VP {
    public PoolVP(int sx) {
        super("sx", sx);
        add("type", "pool");
    }

    public PoolVP sx(int sx) {
        add("sx", sx);
        return this;
    }

    public int sx() {
        int sx = getInt("sx");
        return sx;
    }

    public PoolVP sy(int sy) {
        add("sy", sy);
        return this;
    }

    public PoolVP stride(int stride) {
        add("stride", stride);
        return this;
    }

    public PoolVP pad(int pad) {
        add("pad", pad);
        return this;
    }

    public int sy() {
        int sy = getInt("sy", sx());
        return sy;
    }

    public int stride() {
        int stride = getInt("stride", 2);
        return stride;
    }

    public int pad() {
        int pad = getInt("pad", 0);
        return pad;
    }

    public PoolVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
