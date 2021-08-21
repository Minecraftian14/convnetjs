package in.mcxiv.ai.convnet;

import in.mcxiv.ai.convnet.layers.dotproducts.ConvLayer;
import in.mcxiv.ai.convnet.layers.dotproducts.FullyConnLayer;
import in.mcxiv.ai.convnet.layers.dropout.DropoutLayer;
import in.mcxiv.ai.convnet.layers.input.InputLayer;
import in.mcxiv.ai.convnet.layers.loss.RegressionLayer;
import in.mcxiv.ai.convnet.layers.loss.SVMLayer;
import in.mcxiv.ai.convnet.layers.loss.SoftmaxLayer;
import in.mcxiv.ai.convnet.layers.nonlinearities.MaxoutLayer;
import in.mcxiv.ai.convnet.layers.nonlinearities.ReluLayer;
import in.mcxiv.ai.convnet.layers.nonlinearities.SigmoidLayer;
import in.mcxiv.ai.convnet.layers.nonlinearities.TanhLayer;
import in.mcxiv.ai.convnet.layers.normalization.LocalResponseNormalizationLayer;
import in.mcxiv.ai.convnet.layers.pool.PoolLayer;
import in.mcxiv.ai.convnet.net.Layer;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

public class Net {

    public ArrayList<Layer> layers;

    public Net() {
        this(null);
    }
    /**
     * Net manages a set of layers
     * For now constraints: Simple linear order of layers, first layer input last layer a cost layer
     */
    public Net(Object options) {

        this.layers = new ArrayList();

    }

    public static ArrayList<VP> desugar(ArrayList<VP> defs) {
        ArrayList<VP> new_defs = new ArrayList<>();

        for (VP def : defs) {

            String type = def.getFC("type");

            switch (type) {
                case "softmax":
                case "svm":
                    // set an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    new_defs.add(new VP("type", "fc", "num_neurons", def.get("num_classes")));
                    break;
                case "regression":
                    // set an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    // {type:'fc', num_neurons: def.num_neurons}
                    new_defs.add(new VP("type", "fc", "num_neurons", def.get("num_neurons")));
                    break;
                case "fc":
                case "conv":
                    if (def.isNull("bias_pref")) {
                        def.add("bias_pref", 0.0);
                        if (def.notNull("activation") && def.is("activation", "relu")) {
                            def.set("bias_pref", 0.1);
                            // relus like a bit of positive bias to get gradients early
                            // otherwise it's technically possible that a relu unit will never turn on (by chance)
                            // and will never get any gradient and never contribute any computation. Dead relu.
                        }
                    }
                    break;
            }

            new_defs.add(def);

            String activation = def.getNFC("activation");
            if (def.notNull("activation")) {

                switch (activation) {
                    //@off
                    case "relu"   : new_defs.add(new VP("type", "relu"    ));break;
                    case "sigmoid": new_defs.add(new VP("type", "sigmoid" ));break;
                    case "tanh"   : new_defs.add(new VP("type", "tanh"    ));break;
                    //@on
                    case "maxout":
                        // create maxout activation, and pass along group size, if provided
                        var gs = def.notNull("group_size") ? def.getInt("group_size") : 2;
                        new_defs.add(new VP("type", "maxout", "group_size", gs));
                        break;
                    default:
                        System.err.println("ERROR unsupported activation " + activation);
                }
            }

            if (def.notNull("drop_prob") && !def.is("type", "dropout")) {
                new_defs.add(new VP("type", "dropout", "drop_prob", def.get("drop_prob")));
            }
        }

        return new_defs;
    }

    public void makeLayers(ArrayList<VP> defs) {

        assert defs.size() >= 2 : "Error! At least one input layer and one loss layer are required.";
        assert defs.get(0).is("type", "input") : "Error! First layer must be the input layer, to declare size of inputs";

        // desugar layer_defs for adding activation, dropout layers etc
        defs = desugar(defs);

        // create the layers
        this.layers = new ArrayList<>();
        for (int i = 0, size = defs.size(); i < size; i++) {
            var def = defs.get(i);
            if (i > 0) {
                var prev = this.layers.get(i - 1);
                def.add("in_sx", prev.out_sx);
                def.add("in_sy", prev.out_sy);
                def.add("in_depth", prev.out_depth);
            }

            String type = def.getFC("type");
            switch (type) {
                //@off
                      case "fc":         this.layers.add(new FullyConnLayer (def)); break;
                      case "lrn":        this.layers.add(new LocalResponseNormalizationLayer(def)); break;
                      case "dropout":    this.layers.add(new DropoutLayer   (def)); break;
                      case "input":      this.layers.add(new InputLayer     (def)); break;
                      case "softmax":    this.layers.add(new SoftmaxLayer   (def)); break;
                      case "regression": this.layers.add(new RegressionLayer(def)); break;
                      case "conv":       this.layers.add(new ConvLayer      (def)); break;
                      case "pool":       this.layers.add(new PoolLayer      (def)); break;
                      case "relu":       this.layers.add(new ReluLayer      (def)); break;
                      case "sigmoid":    this.layers.add(new SigmoidLayer   (def)); break;
                      case "tanh":       this.layers.add(new TanhLayer      (def)); break;
                      case "maxout":     this.layers.add(new MaxoutLayer    (def)); break;
                      case "svm":        this.layers.add(new SVMLayer       (def)); break;
                      default: System.err.println("ERROR: UNRECOGNIZED LAYER TYPE: " + type);
                //@on
            }
        }
    }

    public Vol forward(Vol V) {
        return forward(V,false);
    }

    // forward prop the network.
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    public Vol forward(Vol V, boolean is_training) {
        var act = this.layers.get(0).forward(V, is_training);
        for(var i=1;i<this.layers.size();i++) {
            act = this.layers.get(i).forward(act, is_training);
        }
        return act;
    }

    public double getCostLoss(Vol V, Object y) {
        this.forward(V, false);
        var N = this.layers.size();
        var loss = this.layers.get(N-1).backward(y);
        return loss;
    }

    // backprop: compute gradients wrt all parameters
    public double backward(Object y) {
        var N = this.layers.size();
        var loss = this.layers.get(N-1).backward(y); // last layer assumed to be loss layer
        for(var i=N-2;i>=0;i--) { // first layer assumed input
            this.layers.get(i).backward();
        }
        return loss;
    }

    public ArrayList<VP> getParamsAndGrads() {
        // accumulate parameters and gradients for the entire network
        var response = new ArrayList<VP>();
        for(var i=0;i<this.layers.size();i++) {
            var layer_reponse = this.layers.get(i).getParamsAndGrads();
            response.addAll(layer_reponse);
        }
        return response;
    }

    public int getPrediction() {
        // this is a convenience function for returning the argmax
        // prediction, assuming the last layer of the net is a softmax
        var S = this.layers.get(this.layers.size()-1);
        assert S.layer_type.equals("softmax") : "getPrediction function assumes softmax as last layer of the net!";

        var p = S.out_act.w;
        var maxv = p.get(0);
        var maxi = 0;
        for(var i = 1; i<p.size; i++) {
            if(p.get(i) > maxv) { maxv = p.get(i); maxi = i;}
        }
        return maxi; // return index of the class with highest class probability
    }

}























