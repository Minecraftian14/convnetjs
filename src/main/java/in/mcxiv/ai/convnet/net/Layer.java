package in.mcxiv.ai.convnet.net;

import in.mcxiv.ai.convnet.DoubleBuffer;
import in.mcxiv.ai.convnet.Vol;

import java.util.ArrayList;

public abstract class Layer {
    public int in_sx;
    public int in_sy;
    public int in_depth;

    public int out_sx;
    public int out_sy;
    public int out_depth;

    public Vol in_act;
    public Vol out_act;

    // ConvLayer
    public int sx;

    public int sy;
    public int stride;
    public int pad;
    public Double l1_decay_mul;
    public Double l2_decay_mul;
    public String layer_type;

    public ArrayList<Vol> filters;
    public Vol biases;

    // FCL
    public int num_inputs;

    public Layer(VP opt) {
    }

    public abstract Vol forward(Vol V, boolean is_training);

    public void backward() {
    }

    public double backward(Object y) {
        if (y instanceof Integer) {
            Integer i = (Integer) y;
            return backward((int) i);
        } else if (y instanceof DoubleBuffer) {
            DoubleBuffer da = (DoubleBuffer) y;
            if (da.size == 1)
                return backward((int) da.get(0)); // TODO: will round off be better?
            return backward(da);
        } else if (y instanceof DoubleBuffer[]) {
            DoubleBuffer[] das = (DoubleBuffer[]) y;
            return backward(das);
        } else throw new IllegalStateException();
    }

    protected double backward(int y) {
        return 0;
    }

    protected double backward(DoubleBuffer y) {
        return 0;
    }

    protected double backward(DoubleBuffer[] y) {
        return 0;
    }

    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }

}