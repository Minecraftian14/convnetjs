package in.mcxiv.ai.convnet;

import java.util.function.Supplier;

import static java.lang.Math.*;

public class Util {

    // Random number utilities
    private static boolean return_v = false;
    private static double v_val = 0.0f;

    public static double gaussRandom() {
        if (return_v) {
            return_v = false;
            return v_val;
        }
        var u = 2 * random() - 1;
        var v = 2 * random() - 1;
        var r = u * u + v * v;
        if (r == 0 || r > 1) return gaussRandom();
        var c = sqrt(-2 * log(r) / r);
        v_val = v * c; // cache this
        return_v = true;
        return u * c;
    }


    public static double randf(double a, double b) {
        return random() * (b - a) + a;
    }

    // TODO: returns int or double?
    public static int randi(int a, int b) {
        return (int) floor(random() * (b - a) + a);
    }

    // TODO: returns int or double?
    public static double randn(double mu, double std) {
        return mu + gaussRandom() * std;
    }

    // Array utilities
    public static DoubleBuffer zeros(int n) {
        DoubleBuffer array = new DoubleBuffer(n);
        for (int i = 0; i < n; i++) {
            array.set(i,0);
        }
        return array;
    }

    public static DoubleBuffer zeros(double n) {
        if (Double.isNaN(n)) return zeros(0);
        return zeros((int) n);
    }

    public static double[] zeros(Object n) {
        throw new UnsupportedOperationException("There is no 'ArrayBuffer' thingy to support!");

        //    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
        //    if(typeof ArrayBuffer === 'undefined') {
        //      // lacking browser support
        //      var arr = new Array(n);
        //      for(var i=0;i<n;i++) { arr[i]= 0; }
        //      return arr;
        //    } else {
        //      return new Float64Array(n);
        //    }
    }

    public static boolean arrContains(DoubleBuffer arr, double elt) {
        return arr.contains(elt);
    }

    public static DoubleBuffer arrUnique(DoubleBuffer arr) {
        var b = new DoubleBuffer();
        for (int i = 0, n = arr.size; i < n; i++) {
            if (!arrContains(b, arr.get(i))) {
                b.add(arr.get(i));
            }
        }
        return b;
    }

    // return max and min of a given non-empty array.
    public static MaxMinReport maxmin(DoubleBuffer w) {
        // ... ;s
        if (w.size == 0) return null;
        var maxv = w.get(0);
        var minv = w.get(0);
        var maxi = 0;
        var mini = 0;
        var n = w.size;
        for (var i = 1; i < n; i++) {
            if (w.get(i) > maxv) {
                maxv = w.get(i);
                maxi = i;
            }
            if (w.get(i) < minv) {
                minv = w.get(i);
                mini = i;
            }
        }
        return new MaxMinReport(maxi, maxv, mini, minv);
    }

    public static void printAccuracy(Vol[] xi, int[] yi, Net net) {

    }
    public static void printARandomExample(Vol[] xi, int[] yi, Net net) {
        int randomIndex = (int) (xi.length * random());
        var h = net.forward(xi[randomIndex]);
        var y = yi[randomIndex];
        System.out.println("<<<<<<<<<Predictions >>>>>>>>>");
        System.out.println(h.meta());
        double max = -Double.MAX_VALUE;
        int maxi = -1;
        for (int i = 0; i < h.depth; i++) {
            double v = h.get(0, 0, i);
            if (v >= max) {
                max = v;
                maxi = i;
            }
            System.out.printf(" %4.2f ", v);
        }
        System.out.println("\n>>>" + maxi);
        System.out.println("<<<<<<<<<Expectations>>>>>>>>>");
        System.out.println(">>>" + y);
        System.out.println("<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>");
    }

    public static class MaxMinReport {
        public final double maxi;
        public final double maxv;
        public final double mini;
        public final double minv;
        public final double dv;

        public MaxMinReport(double maxi, double maxv, double mini, double minv) {
            this.maxi = maxi;
            this.maxv = maxv;
            this.mini = mini;
            this.minv = minv;
            this.dv = maxv - minv;
        }
    }

    // create random permutation of numbers, in range [0...n-1]
    public static DoubleBuffer randperm(int n) {
        int i = n, j = 0;
        double temp;
        var array = new DoubleBuffer();
        for (var q = 0; q < n; q++) array.set(q, q);
        while (i-- > 0) {
            j = (int) Math.floor(Math.random() * (i + 1));
            array.swap(i, j);
        }
        return array;
    }

    // sample from list lst according to probabilities in list probs
    // the two lists are of same size, and probs adds up to 1
    public static double weightedSample(DoubleBuffer lst, DoubleBuffer probs) {
        var p = randf(0, 1.0);
        var cumprob = 0.0;
        for (int k = 0, n = lst.size; k < n; k++) {
            cumprob += probs.get(k);
            if (p < cumprob) return lst.get(k);
        }
        throw new IllegalStateException("Now now now... How should I handle that?");
    }

    // syntactic sugar function for getting default parameter values
    public static <T> T getopt(T opt, String field_name, T default_value) {
        throw new UnsupportedOperationException();
        //  var getopt = function(opt, field_name, default_value) {
        //    if(typeof field_name === 'string') {
        //      // case of single string
        //      return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
        //    } else {
        //      // assume we are given a list of string instead
        //      var ret = default_value;
        //      for(var i=0;i<field_name.length;i++) {
        //        var f = field_name[i];
        //        if (typeof opt[f] !== 'undefined') {
        //          ret = opt[f]; // overwrite return value
        //        }
        //      }
        //      return ret;
        //    }
        //  }
    }

    public static void __assert__(Supplier<Boolean> condition) {
        __assert__(condition, () -> "Assertion failed");
    }

    public static void __assert__(Supplier<Boolean> condition, Supplier<String> message) {
        if (!condition.get()) {
            String msg = message.get();
            throw new AssertionError(msg);
        }
    }

}
