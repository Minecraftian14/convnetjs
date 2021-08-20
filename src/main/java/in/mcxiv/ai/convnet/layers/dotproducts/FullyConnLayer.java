package in.mcxiv.ai.convnet.layers.dotproducts;

import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class FullyConnLayer extends Layer {


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
        var bias = opt.notNull("bias_pref") ? opt.get("bias_pref") : 0.0;
        this.filters = new ArrayList<>();
        for (var i = 0; i < this.out_depth; i++) {
            this.filters.add(new Vol(1, 1, this.num_inputs));
        }
        this.biases = new Vol(1, 1, this.out_depth, bias);
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        var A = new Vol(1, 1, this.out_depth, 0.0);
        var Vw = V.w;
        for (var i = 0; i < this.out_depth; i++) {
            var a = 0.0;
            var wi = this.filters.get(i).w;
            for (var d = 0; d < this.num_inputs; d++) {
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
        var V = this.in_act;
        V.dw = zeros(V.w.length); // zero out the gradient in input Vol

        // compute gradient wrt weights and data
        for (var i = 0; i < this.out_depth; i++) {
            var tfi = this.filters.get(i);
            var chain_grad = this.out_act.dw.get(i);
            for (var d = 0; d < this.num_inputs; d++) {
                V.dw.addValue(d, tfi.w.get(d) * chain_grad); // grad wrt input data
                tfi.dw.addValue(d, V.w.get(d) * chain_grad); // grad wrt params
            }
            this.biases.dw.addValue(i, chain_grad);
        }
    }
}
