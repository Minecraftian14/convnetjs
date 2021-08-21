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
        if (y instanceof Integer i) {
            return backward((int)i);
        } else if (y instanceof DoubleBuffer da) {
            if (da.size == 1)
                return backward((int) da.get(0)); // TODO: will round off be better?
            return backward(da);
        } else if (y instanceof DoubleBuffer[] das) {
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
        ArrayList<VP> response = new ArrayList<>();
        for (var i = 0; i < this.out_depth; i++) {
            response.add(new VP(
                    "params", this.filters.get(i).w,
                    "grads", this.filters.get(i).dw,
                    "l2_decay_mul", this.l2_decay_mul,
                    "l1_decay_mul", this.l1_decay_mul)
            );
        }
        response.add(new VP(
                "params", this.biases.w,
                "grads", this.biases.dw,
                "l1_decay_mul", 0.0,
                "l2_decay_mul", 0.0
        ));
        return response;
    }

}