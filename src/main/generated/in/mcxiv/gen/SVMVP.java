package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class SVMVP extends VP {
    public SVMVP(int num_classes) {
        super("num_classes", num_classes);
        add("type", "svm");
    }

    public SVMVP num_classes(int num_classes) {
        add("num_classes", num_classes);
        return this;
    }

    public int num_classes() {
        int num_classes = getInt("num_classes");
        return num_classes;
    }

    public SVMVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
