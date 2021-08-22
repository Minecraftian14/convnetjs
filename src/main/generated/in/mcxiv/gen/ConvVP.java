package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VP;
import java.lang.String;

public class ConvVP extends VP {
    public ConvVP(int filters, int sx) {
        super("filters", filters, "sx", sx);
        add("type", "conv");
    }

    public ConvVP filters(int filters) {
        add("filters", filters);
        return this;
    }

    public ConvVP sx(int sx) {
        add("sx", sx);
        return this;
    }

    public int filters() {
        int filters = getInt("filters");
        return filters;
    }

    public int sx() {
        int sx = getInt("sx");
        return sx;
    }

    public ConvVP sy(int sy) {
        add("sy", sy);
        return this;
    }

    public ConvVP stride(int stride) {
        add("stride", stride);
        return this;
    }

    public ConvVP pad(int pad) {
        add("pad", pad);
        return this;
    }

    public ConvVP l1_decay_mul(double l1_decay_mul) {
        add("l1_decay_mul", l1_decay_mul);
        return this;
    }

    public ConvVP l2_decay_mul(double l2_decay_mul) {
        add("l2_decay_mul", l2_decay_mul);
        return this;
    }

    public ConvVP bias_pref(double bias_pref) {
        add("bias_pref", bias_pref);
        return this;
    }

    public int sy() {
        int sy = getInt("sy", sx());
        return sy;
    }

    public int stride() {
        int stride = getInt("stride", 1);
        return stride;
    }

    public int pad() {
        int pad = getInt("pad", 0);
        return pad;
    }

    public double l1_decay_mul() {
        double l1_decay_mul = getD("l1_decay_mul", 0.0);
        return l1_decay_mul;
    }

    public double l2_decay_mul() {
        double l2_decay_mul = getD("l2_decay_mul", 1.0);
        return l2_decay_mul;
    }

    public double bias_pref() {
        double bias_pref = getD("bias_pref", 0.0);
        return bias_pref;
    }

    public ConvVP activation(String activation) {
        add("activation", activation);
        return this;
    }
}
