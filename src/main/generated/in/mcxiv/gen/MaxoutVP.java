package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class MaxoutVP extends VP {
    public MaxoutVP(int group_size) {
        super("group_size", group_size);
        add("type", "maxout");
    }

    public MaxoutVP group_size(int group_size) {
        add("group_size", group_size);
        return this;
    }

    public int group_size() {
        int group_size = getInt("group_size");
        return group_size;
    }

    public MaxoutVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
