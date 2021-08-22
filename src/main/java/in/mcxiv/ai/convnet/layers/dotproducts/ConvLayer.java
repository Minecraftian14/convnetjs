package in.mcxiv.ai.convnet.layers.dotproducts;

import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.annotations.LayerConstructor;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class ConvLayer extends DotProductLayer {

    public static final String LAYER_TAG = "conv";

    @LayerConstructor(
            tag = LAYER_TAG,
            required = "int filters, int sx",
            optional = "int sy sx(), int stride 1, int pad 0, double l1_decay_mul 0.0, double l2_decay_mul 1.0, double bias_pref 0.0"
    )
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
        this.pad = opt.notNull("pad") ? opt.getInt("pad") : 0; // amount of 0 padding to set around borders of input volume
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
        double bias = opt.notNull("bias_pref") ? opt.getD("bias_pref") : 0.0;
        this.filters = new ArrayList<>();
        for (int i = 0; i < this.out_depth; i++) {
            filters.add(new Vol(this.sx, this.sy, this.in_depth));
        }
        this.biases = new Vol(1, 1, this.out_depth, bias);
    }


    public Vol forward(Vol V, boolean is_training) {
        // optimized code by @mdda that achieves 2x speedup over previous version

        this.in_act = V;
        Vol A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

        int V_sx = V.sx;
        int V_sy = V.sy;
        int xy_stride = this.stride;

        for (int d = 0; d < this.out_depth; d++) {
            Vol f = this.filters.get(d);
            int x = -this.pad;
            int y = -this.pad;
            for (int ay = 0; ay < this.out_sy; y += xy_stride, ay++) {  // xy_stride
                x = -this.pad;
                for (int ax = 0; ax < this.out_sx; x += xy_stride, ax++) {  // xy_stride

                    // convolve centered at this particular location
                    double a = 0.0;
                    for (int fy = 0; fy < f.sy; fy++) {
                        int oy = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < f.sx; fx++) {
                            int ox = x + fx;
                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (int fd = 0; fd < f.depth; fd++) {
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

        Vol V = this.in_act;
        V.dw = zeros(V.w.size); // zero out gradient wrt bottom data, we're about to fill it

        int V_sx = V.sx;
        int V_sy = V.sy;
        int xy_stride = this.stride;

        for (int d = 0; d < this.out_depth; d++) {
            Vol f = this.filters.get(d);
            int x = -this.pad;
            int y = -this.pad;
            for (int ay = 0; ay < this.out_sy; y += xy_stride, ay++) {  // xy_stride
                x = -this.pad;
                for (int ax = 0; ax < this.out_sx; x += xy_stride, ax++) {  // xy_stride

                    // convolve centered at this particular location
                    double chain_grad = this.out_act.get_grad(ax, ay, d); // gradient from above, from chain rule
                    for (int fy = 0; fy < f.sy; fy++) {
                        int oy = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < f.sx; fx++) {
                            int ox = x + fx;
                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (int fd = 0; fd < f.depth; fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    int ix1 = ((V_sx * oy) + ox) * V.depth + fd;
                                    int ix2 = ((f.sx * fy) + fx) * f.depth + fd;
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
