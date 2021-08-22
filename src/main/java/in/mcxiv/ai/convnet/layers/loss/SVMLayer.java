package in.mcxiv.ai.convnet.layers.loss;

import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.annotations.VPConstructor;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class SVMLayer extends Layer {

    public static final String LAYER_TAG = "svm";

    @VPConstructor(
            tag = LAYER_TAG,
            required = "int num_classes"
    )
    public SVMLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // computed
        this.num_inputs = opt.getInt("in_sx") * opt.getInt("in_sy") * opt.getInt("in_depth");
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = "svm";
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        this.out_act = V; // nothing to do, output raw scores
        return V;
    }

    @Override
    protected double backward(int y) {
        // compute and accumulate gradient wrt weights and bias of this layer
        Vol x = this.in_act;
        x.dw = zeros(x.w.size); // zero out the gradient of input Vol

        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        double yscore = x.w.get(y); // score of ground truth
        double margin = 1.0;
        double loss = 0.0;
        for (int i = 0; i < this.out_depth; i++) {
            if (y == i) {
                continue;
            }
            double ydiff = -yscore + x.w.get(i) + margin;
            if (ydiff > 0) {
                // violating dimension, apply loss
                x.dw.set(i, 1);
                x.dw.set(y, 1);
                loss += ydiff;
            }
        }

        return loss;
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }
}
