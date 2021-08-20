package in.mcxiv.ai.convnet.layers.dotproducts;

import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class ConvLayer extends Layer {

    public ConvLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // required
        this.out_depth = opt.getInt("filters");
        this.sx = opt.getInt("sx"); // filter size. Should be odd if possible, it's cleaner.
        this.in_depth = opt.getInt("in_depth");
        this.in_sx = opt.getInt("in_sx");
        this.in_sy = opt.getInt("in_sy");

        // optional
        this.sy = opt.notNull("sy") ? opt.getInt("sy") : this.sx;
        this.stride = opt.notNull("stride") ? opt.getInt("stride") : 1; // stride at which we apply filters to input volume
        this.pad = opt.notNull("pad") ? opt.getInt("pad") : 0; // amount of 0 padding to add around borders of input volume
        this.l1_decay_mul = opt.notNull("l1_decay_mul") ? opt.getD("l1_decay_mul") : 0.0;
        this.l2_decay_mul = opt.notNull("l2_decay_mul") ? opt.getD("l2_decay_mul") : 1.0;

        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        this.out_sx = (int) Math.floor((this.in_sx + this.pad * 2 - this.sx) * 1d / this.stride + 1);
        this.out_sy = (int) Math.floor((this.in_sy + this.pad * 2 - this.sy) * 1d / this.stride + 1);
        this.layer_type = "conv";

        // initializations
        var bias = opt.notNull("bias_pref") ? opt.getD("bias_pref") : 0.0;
        this.filters = new ArrayList<>();
        for (int i = 0; i < this.out_depth; i++) {
            filters.add(new Vol(this.sx, this.sy, this.in_depth));
        }
        this.biases = new Vol(1, 1, this.out_depth, bias);
    }


    public Vol forward(Vol V, boolean is_training) {
        // optimized code by @mdda that achieves 2x speedup over previous version

        this.in_act = V;
        var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

        var V_sx = V.sx;
        var V_sy = V.sy;
        var xy_stride = this.stride;

        for (var d = 0; d < this.out_depth; d++) {
            var f = this.filters.get(d);
            var x = -this.pad;
            var y = -this.pad;
            for (var ay = 0; ay < this.out_sy; y += xy_stride, ay++) {  // xy_stride
                x = -this.pad;
                for (var ax = 0; ax < this.out_sx; x += xy_stride, ax++) {  // xy_stride

                    // convolve centered at this particular location
                    var a = 0.0;
                    for (var fy = 0; fy < f.sy; fy++) {
                        var oy = y + fy; // coordinates in the original input array coordinates
                        for (var fx = 0; fx < f.sx; fx++) {
                            var ox = x + fx;
                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (var fd = 0; fd < f.depth; fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += f.w.get(((f.sx * fy) + fx) * f.depth + fd) * V.w.get(((V_sx * oy) + ox) * V.depth + fd);
                                }
                            }
                        }
                    }
                    a += this.biases.w.get(d);
                    A.set(ax, ay, d, a);
                }
            }
        }
        this.out_act = A;
        return this.out_act;
    }

    public void backward() {

        var V = this.in_act;
        V.dw = zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

        var V_sx = V.sx;
        var V_sy = V.sy;
        var xy_stride = this.stride;

        for (var d = 0; d < this.out_depth; d++) {
            var f = this.filters.get(d);
            var x = -this.pad;
            var y = -this.pad;
            for (var ay = 0; ay < this.out_sy; y += xy_stride, ay++) {  // xy_stride
                x = -this.pad;
                for (var ax = 0; ax < this.out_sx; x += xy_stride, ax++) {  // xy_stride

                    // convolve centered at this particular location
                    var chain_grad = this.out_act.get_grad(ax, ay, d); // gradient from above, from chain rule
                    for (var fy = 0; fy < f.sy; fy++) {
                        var oy = y + fy; // coordinates in the original input array coordinates
                        for (var fx = 0; fx < f.sx; fx++) {
                            var ox = x + fx;
                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (var fd = 0; fd < f.depth; fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    var ix1 = ((V_sx * oy) + ox) * V.depth + fd;
                                    var ix2 = ((f.sx * fy) + fx) * f.depth + fd;
                                    f.dw.addValue(ix2, V.w.get(ix1) * chain_grad);
                                    V.dw.addValue(ix1, f.w.get(ix2) * chain_grad);
                                }
                            }
                        }
                    }
                    this.biases.dw.addValue(d, chain_grad);
                }
            }
        }
    }
}
