package in.mcxiv.ai.convnet.layers.dotproducts;

import in.mcxiv.ai.convnet.DoubleBuffer;
import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.annotations.LayerConstructor;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class FullyConnLayer extends DotProductLayer {

    public static final String LAYER_TAG = "fc";

    @LayerConstructor(
            tag = LAYER_TAG,
            required = "int num_neurons",
            optional = "double l1_decay_mul 0.0, double l2_decay_mul 1.0, double bias_pref 0.0"
    )
    public FullyConnLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // required
        // ok fine we will allow 'filters' as the word as well
        this.out_depth = opt.notNull("num_neurons") ? opt.getInt("num_neurons") : opt.getInt("filters");

        // optional
        this.l1_decay_mul = opt.notNull("l1_decay_mul") ? opt.getD("l1_decay_mul") : 0.0;
        this.l2_decay_mul = opt.notNull("l2_decay_mul") ? opt.getD("l2_decay_mul") : 1.0;

        // computed
        this.num_inputs = opt.getInt("in_sx") * opt.getInt("in_sy") * opt.getInt("in_depth");
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = "fc";

        // initializations
        Object bias = opt.notNull("bias_pref") ? opt.get("bias_pref") : 0.0;
        this.filters = new ArrayList<>();
        for (int i = 0; i < this.out_depth; i++) {
            this.filters.add(new Vol(1, 1, this.num_inputs));
        }
        this.biases = new Vol(1, 1, this.out_depth, bias);
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        Vol A = new Vol(1, 1, this.out_depth, 0.0);
        DoubleBuffer Vw = V.w;
        for (int i = 0; i < this.out_depth; i++) {
            double a = 0.0;
            DoubleBuffer wi = this.filters.get(i).w;
            for (int d = 0; d < this.num_inputs; d++) {
                a += Vw.get(d) * wi.get(d); // for efficiency use Vols directly for now
            }
            a += this.biases.w.get(i);
            A.w.set(i, a);
        }
        this.out_act = A;
        return this.out_act;
    }

    @Override
    public void backward() {
        Vol V = this.in_act;
        V.dw = zeros(V.w.size); // zero out the gradient in input Vol

        // compute gradient wrt weights and data
        for (int i = 0; i < this.out_depth; i++) {
            Vol tfi = this.filters.get(i);
            double chain_grad = this.out_act.dw.get(i);
            for (int d = 0; d < this.num_inputs; d++) {
                V.dw.addValue(d, tfi.w.get(d) * chain_grad); // grad wrt input data
                tfi.dw.addValue(d, V.w.get(d) * chain_grad); // grad wrt params
            }
            this.biases.dw.addValue(i, chain_grad);
        }
    }
}
