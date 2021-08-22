package in.mcxiv.ai.convnet.layers.dropout;

import in.mcxiv.ai.convnet.DoubleBuffer;
import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.annotations.VPConstructor;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class DropoutLayer extends Layer {

    public static final String LAYER_TAG = "dropout";

    public DoubleBuffer dropped;
    public double drop_prob;

    @VPConstructor(
            tag = LAYER_TAG,
            optional = "double drop_prob 0.5"
    )
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
        Vol V2 = V.clone();
        int N = V.w.size;
        if (is_training) {
            // do dropout
            for (int i = 0; i < N; i++) {
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
            for (int i = 0; i < N; i++) {
                V2.w.set(i, V2.w.get(i) * this.drop_prob);
            }
        }
        this.out_act = V2;
        return this.out_act; // dummy identity function for now
    }

    @Override
    public void backward() {
        Vol V = this.in_act; // we need to set dw of this
        Vol chain_grad = this.out_act;
        int N = V.w.size;
        V.dw = zeros(N); // zero out gradient wrt data
        for(int i = 0; i<N; i++) {
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
