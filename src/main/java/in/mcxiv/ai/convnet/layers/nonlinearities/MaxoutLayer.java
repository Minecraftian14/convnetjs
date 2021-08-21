package in.mcxiv.ai.convnet.layers.nonlinearities;

import in.mcxiv.ai.convnet.DoubleBuffer;
import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class MaxoutLayer extends Layer {

    public DoubleBuffer switches;
    public int group_size;

    public MaxoutLayer(VP opt) {
        super(opt);

        if (opt == null) opt = new VP();

        // required
        this.group_size = opt.notNull("group_size") ? opt.getInt("group_size") : 2;

        // computed
        this.out_sx = opt.getInt("in_sx");
        this.out_sy = opt.getInt("in_sy");
        this.out_depth = (int) Math.floor(opt.getInt("in_depth") * 1d / this.group_size);
        this.layer_type = "sigmoid";

        this.switches = zeros(this.out_sx * this.out_sy * this.out_depth); // useful for backprop
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        var N = this.out_depth;
        var V2 = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if (this.out_sx == 1 && this.out_sy == 1) {
            for (var i = 0; i < N; i++) {
                var ix = i * this.group_size; // base index offset
                var a = V.w.get(ix);
                var ai = 0;
                for (var j = 1; j < this.group_size; j++) {
                    var a2 = V.w.get(ix + j);
                    if (a2 > a) {
                        a = a2;
                        ai = j;
                    }
                }
                V2.w.set(i, a);
                this.switches.set(i, ix + ai);
                ;
            }
        } else {
            var n = 0; // counter for switches
            for (var x = 0; x < V.sx; x++) {
                for (var y = 0; y < V.sy; y++) {
                    for (var i = 0; i < N; i++) {
                        var ix = i * this.group_size;
                        var a = V.get(x, y, ix);
                        var ai = 0;
                        for (var j = 1; j < this.group_size; j++) {
                            var a2 = V.get(x, y, ix + j);
                            if (a2 > a) {
                                a = a2;
                                ai = j;
                            }
                        }
                        V2.set(x, y, i, a);
                        this.switches.set(n, ix + ai);
                        n++;
                    }
                }
            }

        }
        this.out_act = V2;
        return this.out_act;
    }

    @Override
    public void backward() {
        var V = this.in_act; // we need to set dw of this
        var V2 = this.out_act;
        var N = this.out_depth;
        V.dw = zeros(V.w.size); // zero out gradient wrt data

        // pass the gradient through the appropriate switch
        if(this.out_sx == 1 && this.out_sy ==1) {
            for(var i=0;i<N;i++) {
                var chain_grad = V2.dw.get(i);
                V.dw.set((int)this.switches.get(i), chain_grad);
            }
        } else {
            // bleh okay, lets do this the hard way
            var n=0; // counter for switches
            for(var x=0;x<V2.sx;x++) {
                for(var y=0;y<V2.sy;y++) {
                    for(var i=0;i<N;i++) {
                        var chain_grad = V2.get_grad(x,y,i);
                        V.set_grad(x,y, (int) this.switches.get(n),chain_grad);
                        n++;
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
