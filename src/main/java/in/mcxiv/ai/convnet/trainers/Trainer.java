package in.mcxiv.ai.convnet.trainers;

import in.mcxiv.ai.convnet.DoubleBuffer;
import in.mcxiv.ai.convnet.Net;
import in.mcxiv.ai.convnet.Vol;
import in.mcxiv.ai.convnet.net.VP;

import java.util.ArrayList;

import static in.mcxiv.ai.convnet.Util.zeros;

public class Trainer {

    public Net net;

    public double learning_rate;
    public double l1_decay;
    public double l2_decay;
    public int batch_size;
    public String method;

    public double momentum;
    public double ro;
    public double eps;
    public double beta1;
    public double beta2;

    public int k;
    public ArrayList<DoubleBuffer> gsum;
    public ArrayList<DoubleBuffer> xsum;

    public boolean regression;

    public Trainer(Net net, Object... options) {
        this(net, new VP(options));
    }

    public Trainer(Net net, VP options) {
        if (options == null) options = new VP();

        this.net = net;

        this.learning_rate = options.notNull("learning_rate") ? options.getD("learning_rate") : 0.01;
        this.l1_decay = options.notNull("l1_decay") ? options.getD("l1_decay") : 0.0;
        this.l2_decay = options.notNull("l2_decay") ? options.getD("l2_decay") : 0.0;
        this.batch_size = options.notNull("batch_size") ? options.getInt("batch_size") : 1;
        this.method = options.notNull("method") ? options.getSt("method") : "sgd"; // sgd/adam/adagrad/adadelta/windowgrad/netsterov

        this.momentum = options.notNull("momentum") ? options.getD("momentum") : 0.9;
        this.ro = options.notNull("ro") ? options.getD("ro") : 0.95; // used in adadelta
        this.eps = options.notNull("eps") ? options.getD("eps") : 1e-8; // used in adam or adadelta
        this.beta1 = options.notNull("beta1") ? options.getD("beta1") : 0.9; // used in adam
        this.beta2 = options.notNull("beta2") ? options.getD("beta2") : 0.999; // used in adam

        this.k = 0; // iteration counter
        this.gsum = new ArrayList<>(); // last iteration gradients (used for momentum calculations)
        this.xsum = new ArrayList<>(); // used in adam or adadelta

        // check if regression is expected
        if (this.net.layers.get(this.net.layers.size() - 1).layer_type.equals("regression"))
            this.regression = true;
        else
            this.regression = false;

    }

    public VP train(Vol x, Object y) {

        long start = System.nanoTime();
        this.net.forward(x, true); // also set the flag that lets the net know we're just training
        long end = System.nanoTime();
        long fwd_time = end - start;

        start = System.nanoTime();
        double cost_loss = this.net.backward(y);
        double l2_decay_loss = 0.0;
        double l1_decay_loss = 0.0;
        end = System.nanoTime();
        long bwd_time = end - start;

        if (this.regression && !(y instanceof DoubleBuffer))
            System.err.println("Warning: a regression net requires an array as training output vector.");

        this.k++;
        if (this.k % this.batch_size == 0) {

            ArrayList<VP> pglist = this.net.getParamsAndGrads();

            boolean isNotVanillaSGD = !this.method.equals("sgd") || this.momentum > 0.0;
            // initialize lists for accumulators. Will only be done once on first iteration
            if (this.gsum.size() == 0 && isNotVanillaSGD) {
                // only vanilla sgd doesnt need either lists
                // momentum needs gsum
                // adagrad needs gsum
                // adam and adadelta needs gsum and xsum
                for (int i = 0; i < pglist.size(); i++) {
                    DoubleBuffer params = pglist.get(i).getFC("params");
                    this.gsum.add(zeros(params.size));
                    if (this.method.equals("adam") || this.method.equals("adadelta")) {
                        this.xsum.add(zeros(params.size));
                    } else {
                        this.xsum.add(new DoubleBuffer()); // conserve memory
                    }
                }
            }

            // perform an update for all sets of weights
            for (int i = 0; i < pglist.size(); i++) {
                VP pg = pglist.get(i); // param, gradient, other options in future (custom learning rate etc)
                DoubleBuffer p = pg.getFC("params");
                DoubleBuffer g = pg.getFC("grads");

                // learning rate for some parameters.
                double l2_decay_mul = pg.notNull("l2_decay_mul") ? pg.getD("l2_decay_mul") : 1.0;
                double l1_decay_mul = pg.notNull("l1_decay_mul") ? pg.getD("l1_decay_mul") : 1.0;
                double l2_decay = this.l2_decay * l2_decay_mul;
                double l1_decay = this.l1_decay * l1_decay_mul;

                int plen = p.size;
                for (int j = 0; j < plen; j++) {
                    l2_decay_loss += l2_decay * p.get(j) * p.get(j) / 2; // accumulate weight decay loss
                    l1_decay_loss += l1_decay * Math.abs(p.get(j));
                    double l1grad = l1_decay * (p.get(j) > 0 ? 1 : -1);
                    double l2grad = l2_decay * (p.get(j));

                    double gij = (l2grad + l1grad + g.get(j)) / this.batch_size; // raw batch gradient

                    DoubleBuffer gsumi = null;
                    DoubleBuffer xsumi = null;
                    if (isNotVanillaSGD) {
                        gsumi = this.gsum.get(i);
                        xsumi = this.xsum.get(i);
                    }
                    if (this.method.equals("adam")) {
                        // adam update
                        gsumi.set(j, gsumi.get(j) * this.beta1 + (1 - this.beta1) * gij); // update biased first moment estimate
                        xsumi.set(j, xsumi.get(j) * this.beta2 + (1 - this.beta2) * gij * gij); // update biased second moment estimate
                        double biasCorr1 = gsumi.get(j) * (1 - Math.pow(this.beta1, this.k)); // correct bias first moment estimate
                        double biasCorr2 = xsumi.get(j) * (1 - Math.pow(this.beta2, this.k)); // correct bias second moment estimate
                        double dx = -this.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
                        p.addValue(j, dx);
                    } else if (this.method.equals("adagrad")) {
                        // adagrad update
                        gsumi.set(j, gsumi.get(j) + gij * gij);
                        double dx = -this.learning_rate / Math.sqrt(gsumi.get(j) + this.eps) * gij;
                        p.addValue(j, dx);
                    } else if (this.method.equals("windowgrad")) {
                        // this is adagrad but with a moving window weighted average
                        // so the gradient is not accumulated over the entire history of the run.
                        // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                        gsumi.set(j, this.ro * gsumi.get(j) + (1 - this.ro) * gij * gij);
                        double dx = -this.learning_rate / Math.sqrt(gsumi.get(j) + this.eps) * gij; // eps added for better conditioning
                        p.addValue(j, dx);
                    } else if (this.method.equals("adadelta")) {
                        gsumi.set(j, this.ro * gsumi.get(j) + (1 - this.ro) * gij * gij);
                        double dx = -Math.sqrt((xsumi.get(j) + this.eps) / (gsumi.get(j) + this.eps)) * gij;
                        xsumi.set(j, this.ro * xsumi.get(j) + (1 - this.ro) * dx * dx); // yes, xsum lags behind gsum by 1.
                        p.addValue(j, dx);
                    } else if (this.method.equals("nesterov")) {
                        double dx = gsumi.get(j);
                        gsumi.set(j, gsumi.get(j) * this.momentum + this.learning_rate * gij);
                        dx = this.momentum * dx - (1.0 + this.momentum) * gsumi.get(j);
                        p.addValue(j, dx);
                    } else {
                        // assume SGD
                        if (this.momentum > 0.0) {
                            // momentum update
                            double dx = this.momentum * gsumi.get(j) - this.learning_rate * gij; // step
                            gsumi.set(j, dx); // back this up for next iteration of momentum
                            p.addValue(j, dx);// apply corrected gradient
                        } else {
                            // vanilla sgd
                            p.addValue(j, -this.learning_rate * gij);
                        }
                    }
                    g.set(j, 0); // zero out gradient so that we can begin accumulating anew
                }
            }
        }

        // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
        // in future, TODO: have to completely redo the way loss is done around the network as currently
        // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
        // and it should all be computed correctly and automatically.
        return new VP("fwd_time", fwd_time, "bwd_time", bwd_time,
                "l2_decay_loss", l2_decay_loss, "l1_decay_loss", l1_decay_loss,
                "cost_loss", cost_loss, "softmax_loss", cost_loss,
                "loss", (cost_loss + l1_decay_loss + l2_decay_loss));
    }
}

