package in.mcxiv.ai.convnet.layers.normalization;

import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class LocalResponseNormalizationLayer extends Layer {

    public int k;
    public int n;
    public double alpha;
    public double beta;
    private Vol S_cache_;

    public LocalResponseNormalizationLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // required
        this.k = opt.getInt("k");
        this.n = opt.getInt("n");
        this.alpha = opt.getD("alpha");
        this.beta = opt.getD("beta");

        // computed
        this.out_sx = opt.getInt("in_sx");
        this.out_sy = opt.getInt("in_sy");
        this.out_depth = opt.getInt("in_depth");
        this.layer_type = "lrn";

        // checks
        if (this.n % 2 == 0) {
            System.err.println("WARNING n should be odd for LRN layer");
        }
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;

        var A = V.cloneAndZero();
        this.S_cache_ = V.cloneAndZero();
        var n2 = Math.floor(this.n / 2d);
        for (var x = 0; x < V.sx; x++) {
            for (var y = 0; y < V.sy; y++) {
                for (var i = 0; i < V.depth; i++) {

                    var ai = V.get(x, y, i);

                    // normalize in a window of size n
                    var den = 0.0;
                    for (int j = (int) Math.max(0, i - n2); j <= Math.min(i + n2, V.depth - 1); j++) {
                        var aa = V.get(x, y, j);
                        den += aa * aa;
                    }
                    den *= this.alpha / this.n;
                    den += this.k;
                    this.S_cache_.set(x, y, i, den); // will be useful for backprop
                    den = Math.pow(den, this.beta);
                    A.set(x, y, i, ai / den);
                }
            }
        }

        this.out_act = A;
        return this.out_act; // dummy identity function for now
    }

    @Override
    public void backward() {
        // evaluate gradient wrt data
        var V = this.in_act; // we need to set dw of this
        V.dw = zeros(V.w.length); // zero out gradient wrt data
        var A = this.out_act; // computed in forward pass

        var n2 = Math.floor(this.n / 2d);
        for (var x = 0; x < V.sx; x++) {
            for (var y = 0; y < V.sy; y++) {
                for (var i = 0; i < V.depth; i++) {

                    var chain_grad = this.out_act.get_grad(x, y, i);
                    var S = this.S_cache_.get(x, y, i);
                    var SB = Math.pow(S, this.beta);
                    var SB2 = SB * SB;

                    // normalize in a window of size n
                    for (int j = (int) Math.max(0, i - n2); j <= Math.min(i + n2, V.depth - 1); j++) {
                        var aj = V.get(x, y, j);
                        var g = -aj * this.beta * Math.pow(S, this.beta - 1) * this.alpha / this.n * 2 * aj;
                        if (j == i) g += SB;
                        g /= SB2;
                        g *= chain_grad;
                        V.add_grad(x, y, j, g);
                    }

                }
            }
        }
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }
}
