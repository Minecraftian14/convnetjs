package in.mcxiv.ai.convnet.layers.input;

import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.annotations.VPConstructor;

import java.util.ArrayList;

public class InputLayer extends Layer {

    public static final String LAYER_TAG = "input";

    @VPConstructor(
            tag = LAYER_TAG,
            required = "int out_depth",
            optional = "int out_sx 1, int out_sy 1"
    )
    public InputLayer(VP opt) {
        super(opt);
        if (opt == null) opt = new VP();

        // required: depth
        this.out_depth = opt.notNull("out_depth") ? opt.getInt("out_depth") :
                opt.notNull("depth") ? opt.getInt("depth") : 0;

        // optional: default these dimensions to 1
        this.out_sx = opt.notNull("out_sx") ? opt.getInt("out_sx") :
                opt.notNull("sx") ? opt.getInt("sx") :
                        opt.notNull("width") ? opt.getInt("width") : 1;
        this.out_sy = opt.notNull("out_sy") ? opt.getInt("out_sy") :
                opt.notNull("sy") ? opt.getInt("sy") :
                        opt.notNull("height") ? opt.getInt("height") : 1;

        // computed
        this.layer_type = "input";
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        this.out_act = V;
        return this.out_act; // simply identity function for now
    }

    @Override
    public void backward() {
    }

    @Override
    public ArrayList<VP> getParamsAndGrads() {
        return new ArrayList<>();
    }

}
