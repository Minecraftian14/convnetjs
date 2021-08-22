package in.mcxiv.ai.convnet;

import in.mcxiv.ai.convnet.net.VP;
import in.mcxiv.ai.convnet.net.VPL;
import in.mcxiv.ai.convnet.trainers.Trainer;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.DoubleAdder;

import static in.mcxiv.ai.convnet.Util.*;

/**
 * A MagicNet takes data: a list of convnetjs.Vol(), and labels
 * which for now are assumed to be class indeces 0..K. MagicNet then:
 * - creates data folds for cross-validation
 * - samples candidate networks
 * - evaluates candidate networks on all data folds
 * - produces predictions by model-averaging the best networks
 */
public class MagicNet {

    public Vol[] data;
    public int[] labels;

    public double train_ratio;
    public int num_folds;
    public int num_candidates;
    public int num_epochs;
    public int ensemble_size;

    public int batch_size_min;
    public int batch_size_max;
    public double l2_decay_min;
    public double l2_decay_max;
    public double learning_rate_min;
    public double learning_rate_max;
    public double momentum_min;
    public double momentum_max;
    public int neurons_min;
    public int neurons_max;

    public VPL folds;
    public ArrayList<VP> candidates;
    public List<VP> evaluated_candidates;
    public int[] unique_labels;
    public int iter;
    public int foldix;
    public Runnable finish_fold_callback;
    public Runnable finish_batch_callback;

    public MagicNet(Vol[] data, int[] labels, VP opt) {
        if (opt == null) opt = new VP();

        if (data == null) data = new Vol[0];
        if (labels == null) labels = new int[0];

        // required inputs
        this.data = data; // store these pointers to data
        this.labels = labels;

        // optional inputs
        this.train_ratio = opt.getD("train_ratio", 0.7);
        this.num_folds = opt.getInt("num_folds", 10);
        this.num_candidates = opt.getInt("num_candidates", 50); // we evaluate several in parallel
        // how many epochs of data to train every network? for every fold?
        // higher values mean higher accuracy in final results, but more expensive
        this.num_epochs = opt.getInt("num_epochs", 50);
        // number of best models to average during prediction. Usually higher = better
        this.ensemble_size = opt.getInt("ensemble_size", 10);

        // candidate parameters
        this.batch_size_min = opt.getInt("batch_size_min", 10);
        this.batch_size_max = opt.getInt("batch_size_max", 300);
        this.l2_decay_min = opt.getD("l2_decay_min", -4);
        this.l2_decay_max = opt.getD("l2_decay_max", 2);
        this.learning_rate_min = opt.getD("learning_rate_min", -4);
        this.learning_rate_max = opt.getD("learning_rate_max", 0);
        this.momentum_min = opt.getD("momentum_min", 0.9);
        this.momentum_max = opt.getD("momentum_max", 0.9);
        this.neurons_min = opt.getInt("neurons_min", 5);
        this.neurons_max = opt.getInt("neurons_max", 30);

        // computed
        this.folds = new VPL(); // data fold indices, gets filled by sampleFolds()
        this.candidates = new ArrayList<>(); // candidate networks that are being currently evaluated
        this.evaluated_candidates = new ArrayList<>(); // history of all candidates that were fully evaluated on all folds
        this.unique_labels = arrUnique(labels);
        this.iter = 0; // iteration counter, goes from 0 -> num_epochs * num_training_data
        this.foldix = 0; // index of active fold

        // callbacks
        this.finish_fold_callback = null;
        this.finish_batch_callback = null;

        // initializations
        if (this.data.length > 0) {
            this.sampleFolds();
            this.sampleCandidates();
        }

    }

    public void sampleFolds() {
        int N = this.data.length;
        int num_train = (int) Math.floor(this.train_ratio * N);
        this.folds.clear(); // flush folds, if any
        for (int i = 0; i < this.num_folds; i++) {
            DoubleBuffer p = randperm(N);
            this.folds.push("train_ix", p.slice(0, num_train), "test_ix", p.slice(num_train, N));
        }
    }

    static final String[] actnms = new String[]{"tanh", "maxout", "relu"};

    public VP sampleCandidate() {
        int input_depth = this.data[0].w.size;
        int num_classes = this.unique_labels.length;

        // sample network topology and hyperparameters
        VPL layer_defs = new VPL();
        layer_defs.push("type", "input", "out_sx", 1, "out_sy", 1, "out_depth", input_depth);
        double nl = weightedSample(new DoubleBuffer(0., 1., 2., 3.), new DoubleBuffer(0.2, 0.3, 0.3, 0.2)); // prefer nets with 1,2 hidden layers
        for (int q = 0; q < nl; q++) {
            int ni = randi(this.neurons_min, this.neurons_max);
            String act = actnms[randi(0, 3)];
            if (randf(0, 1) < 0.5) {
                double dp = Math.random();
                layer_defs.push("type", "fc", "num_neurons", ni, "activation", act, "drop_prob", dp);
            } else {
                layer_defs.push("type", "fc", "num_neurons", ni, "activation", act);
            }
        }
        layer_defs.push("type", "softmax", "num_classes", num_classes);
        Net net = new Net();
        net.makeLayers(layer_defs);

        // sample training hyperparameters
        int bs = randi(this.batch_size_min, this.batch_size_max); // batch size
        double l2 = Math.pow(10, randf(this.l2_decay_min, this.l2_decay_max)); // l2 weight decay
        double lr = Math.pow(10, randf(this.learning_rate_min, this.learning_rate_max)); // learning rate
        double mom = randf(this.momentum_min, this.momentum_max); // momentum. Lets just use 0.9, works okay usually ;p
        double tp = randf(0, 1); // trainer type
        VP trainer_def;
        if (tp < 0.33) {
            trainer_def = new VP("method", "adadelta", "batch_size", bs, "l2_decay", l2);
        } else if (tp < 0.66) {
            trainer_def = new VP("method", "adagrad", "learning_rate", lr, "batch_size", bs, "l2_decay", l2);
        } else {
            trainer_def = new VP("method", "sgd", "learning_rate", lr, "momentum", mom, "batch_size", bs, "l2_decay", l2);
        }

        Trainer trainer = new Trainer(net, trainer_def);

        return new VP(
                "acc", new DoubleBuffer(),
                "accv", new DoubleAdder(),
                "layer_defs", layer_defs,
                "trainer_def", trainer_def,
                "net", net,
                "trainer", trainer);
    }

    public void sampleCandidates() {
        this.candidates.clear(); // flush, if any
        for (int i = 0; i < this.num_candidates; i++) {
            VP cand = this.sampleCandidate();
            this.candidates.add(cand);
        }
    }

    public void step() {
        // run an example through current candidate
        this.iter++;

        // step all candidates on a random data point
        VP fold = this.folds.get(this.foldix); // active fold
        DoubleBuffer train_ixDB = fold.getFC("train_ix");
        int dataix = (int) train_ixDB.get(randi(0, train_ixDB.size));
        for (int k = 0; k < this.candidates.size(); k++) {
            Vol x = this.data[dataix];
            int l = this.labels[dataix];
            Trainer trainer = this.candidates.get(k).getFC("trainer");
            trainer.train(x, l);
        }

        // process consequences: sample new folds, or candidates
        int lastiter = this.num_epochs * train_ixDB.size;
        if (this.iter >= lastiter) {
            // finished evaluation of this fold. Get final validation
            // accuracies, record them, and go on to next fold.
            DoubleBuffer val_acc = this.evalValErrors();
            for (int k = 0; k < this.candidates.size(); k++) {
                VP c = this.candidates.get(k);
                DoubleBuffer acc = c.getFC("acc");
                acc.add(val_acc.get(k));
                DoubleAdder accv = c.getFC("accv");
                accv.add(val_acc.get(k));
            }
            this.iter = 0; // reset step number
            this.foldix++; // increment fold

            if (this.finish_fold_callback != null) {
                this.finish_fold_callback.run();
            }

            if (this.foldix >= this.folds.size()) {
                // we finished all folds as well! Record these candidates
                // and sample new ones to evaluate.
                for (int k = 0; k < this.candidates.size(); k++) {
                    this.evaluated_candidates.add(this.candidates.get(k));
                }
                // sort evaluated candidates according to accuracy achieved
                this.evaluated_candidates.sort((vp_a, vp_b) -> {
                    DoubleAdder accv_a = vp_a.getFC("accv");
                    DoubleAdder accv_b = vp_b.getFC("accv");
                    DoubleBuffer acc_a = vp_a.getFC("acc");
                    DoubleBuffer acc_b = vp_b.getFC("acc");
                    double a_a = accv_a.doubleValue() / acc_a.size;
                    double a_b = accv_b.doubleValue() / acc_b.size;
                    return Double.compare(a_b, a_a); // TODO: Double.compare(a_b, a_a) or Double.compare(a_a, a_b)
                });
                // and clip only to the top few ones (lets place limit at 3*ensemble_size)
                // otherwise there are concerns with keeping these all in memory
                // if MagicNet is being evaluated for a very long time
                if (this.evaluated_candidates.size() > 3 * this.ensemble_size) {
                    this.evaluated_candidates = this.evaluated_candidates.subList(0, 3 * this.ensemble_size);
                }
                if (this.finish_batch_callback != null) {
                    this.finish_batch_callback.run();
                }
                this.sampleCandidates(); // begin with new candidates
                this.foldix = 0; // reset this
            } else {
                // we will go on to another fold. reset all candidates nets
                for (int k = 0; k < this.candidates.size(); k++) {
                    VP c = this.candidates.get(k);
                    Net net = new Net();
                    VPL layer_defs = c.getFC("layer_defs");
                    net.makeLayers(layer_defs);
                    VP trainer_def = c.getFC("trainer_def");
                    Trainer trainer = new Trainer(net, trainer_def);
                    c.set("net", net);
                    c.set("trainer", trainer);
                }
            }
        }
    }

    public DoubleBuffer evalValErrors() {
        // evaluate candidates on validation data and return performance of current networks
        // as simple list
        DoubleBuffer vals = new DoubleBuffer();
        VP fold = this.folds.get(this.foldix); // active fold
        for (int k = 0; k < this.candidates.size(); k++) {
            Net net = this.candidates.get(k).getFC("net");
            double v = 0.0;
            DoubleBuffer test_ixDB = fold.getFC("test_ix");
            for (int q = 0; q < test_ixDB.size; q++) {
                Vol x = this.data[(int) test_ixDB.get(q)];
                int l = this.labels[(int) test_ixDB.get(q)];
                net.forward(x);
                int yhat = net.getPrediction();
                v += (yhat == l ? 1.0 : 0.0); // 0 1 loss
            }
            v /= test_ixDB.size; // normalize
            vals.add(v);
        }
        return vals;
    }

}
