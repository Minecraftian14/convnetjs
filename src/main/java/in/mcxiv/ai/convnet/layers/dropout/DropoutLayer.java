package in.mcxiv.ai.convnet.layers.dropout;

import in.mcxiv.ai.convnet.DoubleArray;
import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class DropoutLayer extends Layer {

    public DoubleArray dropped;
    public double drop_prob;

    public DropoutLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // computed
        this.out_sx = opt.getInt("in_sx");
        this.out_sy = opt.getInt("in_sy");
        this.out_depth = opt.getInt("in_depth");
        this.layer_type = "dropout";
        this.drop_prob = opt.notNull("drop_prob") ? opt.getD("drop_prob") : 0.5;
        this.dropped = zeros(this.out_sx * this.out_sy * this.out_depth);
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        if (is_training) {
            // do dropout
            for (var i = 0; i < N; i++) {
                if (Math.random() < this.drop_prob) {
                    V2.w.set(i, 0);
                    this.dropped.set(i, true);
                } // drop!
                else {
                    this.dropped.is(i, false);
                }
            }
        } else {
            // scale the activations during prediction
            for (var i = 0; i < N; i++) {
                V2.w.set(i, V2.w.get(i) * this.drop_prob);
            }
        }
        this.out_act = V2;
        return this.out_act; // dummy identity function for now
    }

    @Override
    public void backward() {
        var V = this.in_act; // we need to set dw of this
        var chain_grad = this.out_act;
        var N = V.w.length;
        V.dw = zeros(N); // zero out gradient wrt data
        for(var i=0;i<N;i++) {
            if(this.dropped.get(i)==0) {
                V.dw.set(i, chain_grad.dw.get(i)); // copy over the gradient
            }
        }
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }
}
