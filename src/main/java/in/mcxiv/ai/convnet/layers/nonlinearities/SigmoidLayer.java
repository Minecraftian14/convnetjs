package in.mcxiv.ai.convnet.layers.nonlinearities;

import in.mcxiv.ai.convnet.DoubleBuffer;
import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.annotations.LayerConstructor;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class SigmoidLayer extends Layer {

    public static final String LAYER_TAG = "sigmoid";

    @LayerConstructor(
            tag = LAYER_TAG
    )
    public SigmoidLayer(VP opt) {
        super(opt);

        if (opt == null) opt = new VP();

        // computed
        this.out_sx = opt.getInt("in_sx");
        this.out_sy = opt.getInt("in_sy");
        this.out_depth = opt.getInt("in_depth");
        this.layer_type = "sigmoid";
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        Vol V2 = V.cloneAndZero();
        int N = V.w.size;
        DoubleBuffer V2w = V2.w;
        DoubleBuffer Vw = V.w;
        for (int i = 0; i < N; i++) {
            V2w.set(i, 1.0 / (1.0 + Math.exp(-Vw.get(i))));
        }
        this.out_act = V2;
        return this.out_act;
    }

    @Override
    public void backward() {
        Vol V = this.in_act; // we need to set dw of this
        Vol V2 = this.out_act;
        int N = V.w.size;
        V.dw = zeros(N); // zero out gradient wrt data
        for (int i = 0; i < N; i++) {
            double v2wi = V2.w.get(i);
            V.dw.set(i, v2wi * (1.0 - v2wi) * V2.dw.get(i));
        }
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }
}
