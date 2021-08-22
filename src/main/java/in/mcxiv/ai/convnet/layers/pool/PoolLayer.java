package in.mcxiv.ai.convnet.layers.pool;

import in.mcxiv.ai.convnet.DoubleBuffer;
import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.annotations.LayerConstructor;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class PoolLayer extends Layer {

    public static final String LAYER_TAG = "pool";

    private final DoubleBuffer switchx;
    private final DoubleBuffer switchy;

    @LayerConstructor(
            tag = LAYER_TAG,
            required = "int sx",
            optional = "int sy sx(), int stride 2, int pad 0"
    )
    public PoolLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // required
        this.sx = opt.getInt("sx"); // filter size
        this.in_depth = opt.getInt("in_depth");
        this.in_sx = opt.getInt("in_sx");
        this.in_sy = opt.getInt("in_sy");

        // optional
        this.sy = opt.notNull("sy") ? opt.getInt("sy") : this.sx;
        this.stride = opt.notNull("stride") ? opt.getInt("stride") : 2;
        this.pad = opt.notNull("pad") ? opt.getInt("pad ") : 0; // amount of 0 padding to set around borders of input volume

        // computed
        this.out_depth = this.in_depth;
        this.out_sx = (int) Math.floor((this.in_sx + this.pad * 2 - this.sx) * 1d / this.stride + 1);
        this.out_sy = (int) Math.floor((this.in_sy + this.pad * 2 - this.sy) * 1d / this.stride + 1);
        this.layer_type = "pool";
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        this.switchx = zeros(this.out_sx * this.out_sy * this.out_depth);
        this.switchy = zeros(this.out_sx * this.out_sy * this.out_depth);
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;

        Vol A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

        int n = 0; // a counter for switches
        for (int d = 0; d < this.out_depth; d++) {
            int x = -this.pad;
            int y = -this.pad;
            for (int ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                y = -this.pad;
                for (int ay = 0; ay < this.out_sy; y += this.stride, ay++) {

                    // convolve centered at this particular location
                    double a = -99999; // hopefully small enough ;\
                    int winx = -1, winy = -1;
                    for (int fx = 0; fx < this.sx; fx++) {
                        for (int fy = 0; fy < this.sy; fy++) {
                            int oy = y + fy;
                            int ox = x + fx;
                            if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                                double v = V.get(ox, oy, d);
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if (v > a) {
                                    a = v;
                                    winx = ox;
                                    winy = oy;
                                }
                            }
                        }
                    }
                    this.switchx.set(n, winx);
                    this.switchy.set(n, winy);
                    n++;
                    A.set(ax, ay, d, a);
                }
            }
        }
        this.out_act = A;
        return this.out_act;
    }

    @Override
    public void backward() {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here
        Vol V = this.in_act;
        V.dw = zeros(V.w.size); // zero out gradient wrt data
        Vol A = this.out_act; // computed in forward pass

        int n = 0;
        for (int d = 0; d < this.out_depth; d++) {
            int x = -this.pad;
            int y = -this.pad;
            for (int ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                y = -this.pad;
                for (int ay = 0; ay < this.out_sy; y += this.stride, ay++) {

                    double chain_grad = this.out_act.get_grad(ax, ay, d);
                    V.add_grad((int) this.switchx.get(n), (int) this.switchy.get(n), d, chain_grad);
                    n++;

                }
            }
        }
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }
}
