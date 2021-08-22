package in.mcxiv.ai.convnet.layers.loss;

import in.mcxiv.ai.convnet.DoubleBuffer;
import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.annotations.LayerConstructor;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class SoftmaxLayer extends Layer {

    public static final String LAYER_TAG = "softmax";

    private DoubleBuffer es;

    @LayerConstructor(
            tag = LAYER_TAG,
            required = "int num_classes"
    )
    public SoftmaxLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // computed
        this.num_inputs = opt.getInt("in_sx") * opt.getInt("in_sy") * opt.getInt("in_depth");
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = "softmax";
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;

        Vol A = new Vol(1, 1, this.out_depth, 0.0);

        // compute max activation
        DoubleBuffer as = V.w;
        double amax = V.w.get(0);
        for (int i = 1; i < this.out_depth; i++) {
            if (as.get(i) > amax) amax = as.get(i);
        }

        // compute exponentials (carefully to not blow up)
        DoubleBuffer es = zeros(this.out_depth);
        double esum = 0.0;
        for (int i = 0; i < this.out_depth; i++) {
            double e = Math.exp(as.get(i) - amax);
            esum += e;
            es.set(i, e);
        }

        // normalize and output to sum to one
        for (int i = 0; i < this.out_depth; i++) {
            es.set(i, es.get(i) / esum);
            A.w.set(i, es.get(i));
        }

        this.es = es; // save these for backprop
        this.out_act = A;
        return this.out_act;
    }

    @Override
    public double backward(int y) {
        // compute and accumulate gradient wrt weights and bias of this layer
        Vol x = this.in_act;
        x.dw = zeros(x.w.size); // zero out the gradient of input Vol

        for(int i = 0; i<this.out_depth; i++) {
            double indicator = i == y ? 1.0 : 0.0;
            double mul = -(indicator - this.es.get(i));
            x.dw.set(i,  mul);
        }

        // loss is the class negative log likelihood
        return -Math.log(this.es.get(y));
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }
}
