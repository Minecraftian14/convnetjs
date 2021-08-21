package in.mcxiv.ai.convnet.layers.nonlinearities;

import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class ReluLayer extends Layer {

    public ReluLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // computed
        this.out_sx = opt.getInt("in_sx");
        this.out_sy = opt.getInt("in_sy");
        this.out_depth = opt.getInt("in_depth");
        this.layer_type = "relu";
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.size;
        var V2w = V2.w;
        for (var i = 0; i < N; i++) {
            if (V2w.get(i) < 0) V2w.set(i, 0); // threshold at 0
        }
        this.out_act = V2;
        return this.out_act;
    }

    @Override
    public void backward() {
        var V = this.in_act; // we need to set dw of this
        var V2 = this.out_act;
        var N = V.w.size;
        V.dw = zeros(N); // zero out gradient wrt data
        for(var i=0;i<N;i++) {
            if(V2.w.get(i) <= 0) V.dw.set(i,0); // threshold
            else V.dw.set(i, V2.dw.get(i));
        }
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }
}
