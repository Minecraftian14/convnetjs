package in.mcxiv.ai.convnet.layers.dotproducts;

import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

public abstract class DotProductLayer extends Layer {

    public DotProductLayer(VP opt) {
        super(opt);
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        ArrayList<VP> response = new ArrayList<>();
        for (int i = 0; i < this.out_depth; i++) {
            response.add(new VP(
                    "params", this.filters.get(i).w,
                    "grads", this.filters.get(i).dw,
                    "l2_decay_mul", this.l2_decay_mul,
                    "l1_decay_mul", this.l1_decay_mul)
            );
        }
        response.add(new VP(
                "params", this.biases.w,
                "grads", this.biases.dw,
                "l1_decay_mul", 0.0,
                "l2_decay_mul", 0.0
        ));
        return response;
    }
}
