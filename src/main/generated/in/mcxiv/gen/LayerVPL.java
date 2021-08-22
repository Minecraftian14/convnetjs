package in.mcxiv.gen;

import static java.time.ZonedDateTime.*;
import static java.util.Date.UTC;

import in.mcxiv.ai.convnet.net.VPL;

public class LayerVPL extends VPL {
    public DropoutVP dropout() {
        DropoutVP vp = new DropoutVP();
        add(vp);
        return vp;
    }

    public FullyConnVP fc(int num_neurons) {
        FullyConnVP vp = new FullyConnVP(num_neurons);
        add(vp);
        return vp;
    }

    public SVMVP svm(int num_classes) {
        SVMVP vp = new SVMVP(num_classes);
        add(vp);
        return vp;
    }

    public SigmoidVP sigmoid() {
        SigmoidVP vp = new SigmoidVP();
        add(vp);
        return vp;
    }

    public ReluVP relu() {
        ReluVP vp = new ReluVP();
        add(vp);
        return vp;
    }

    public PoolVP pool(int sx) {
        PoolVP vp = new PoolVP(sx);
        add(vp);
        return vp;
    }

    public TanhVP tanh() {
        TanhVP vp = new TanhVP();
        add(vp);
        return vp;
    }

    public InputVP input(int out_depth) {
        InputVP vp = new InputVP(out_depth);
        add(vp);
        return vp;
    }

    public LocalResponseNormalizationVP lnr(int k, int n, double alpha, double beta) {
        LocalResponseNormalizationVP vp = new LocalResponseNormalizationVP(k, n, alpha, beta);
        add(vp);
        return vp;
    }

    public SoftmaxVP softmax(int num_classes) {
        SoftmaxVP vp = new SoftmaxVP(num_classes);
        add(vp);
        return vp;
    }

    public RegressionVP regression(int num_neurons) {
        RegressionVP vp = new RegressionVP(num_neurons);
        add(vp);
        return vp;
    }

    public MaxoutVP maxout(int group_size) {
        MaxoutVP vp = new MaxoutVP(group_size);
        add(vp);
        return vp;
    }

    public ConvVP conv(int filters, int sx) {
        ConvVP vp = new ConvVP(filters, sx);
        add(vp);
        return vp;
    }
}
