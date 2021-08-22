package in.mcxiv.ai.convnet.net;

import java.util.ArrayList;

public class VPL extends ArrayList<VP> {

    public boolean push(VP vp) {
        return add(vp);
    }

    public boolean add(Object... args) {
        return super.add(new VP(args));
    }

    public boolean push(Object... args) {
        return add(args);
    }

}
