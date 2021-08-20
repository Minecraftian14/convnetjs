package in.mcxiv.ai.convnet.layers.loss;

import in.mcxiv.ai.convnet.DoubleArray;
import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class RegressionLayer extends Layer {
    public RegressionLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // computed
        this.num_inputs = opt.getInt("in_sx") * opt.getInt("in_sy") * opt.getInt("in_depth");
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = "regression";
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        this.out_act = V;
        return V; // identity function
    }

    @Override
    public double backward(Object inp) {

        // compute and accumulate gradient wrt weights and bias of this layer
        var x = this.in_act;
        x.dw = zeros(x.w.length); // zero out the gradient of input Vol
        var loss = 0.0;

        if (inp instanceof Integer y) {
            // lets hope that only one number is being regressed
            var dy = x.w.get(0) - y;
            x.dw.set(0, dy);
            loss += 0.5*dy*dy;
        } else if (inp instanceof DoubleArray y) {
            for(var i=0;i<this.out_depth;i++) {
                var dy = x.w.get(i) - y.get(i);
                x.dw.set(i, dy);
                loss += 0.5*dy*dy;
            }
        } else if (inp instanceof DoubleArray[] y) {
            throw new IllegalStateException();
//            // assume it is a struct with entries .dim and .val
//            // and we pass gradient only along dimension dim to be equal to val
//            var i = y.dim;
//            var yi = y.val;
//            var dy = x.w[i] - yi;
//            x.dw[i] = dy;
//            loss += 0.5*dy*dy;
        } else throw new IllegalStateException();

        return loss;
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }
}
