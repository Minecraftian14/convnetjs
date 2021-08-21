package in.mcxiv.ai.convnet;

import static in.mcxiv.ai.convnet.Util.randn;
import static in.mcxiv.ai.convnet.Util.zeros;

/**
 * Vol is the basic building block of all data in a net.
 * it is essentially just a 3D volume of numbers, with a
 * width (sx), height (sy), and depth (depth).
 * it is used to hold data for all filters, all volumes,
 * all weights, and also stores all gradients w.r.t.
 * the data. c is optionally a value to initialize the volume
 * with. If c is missing, fills the Vol with random numbers.
 */
public class Vol {

    public int sx;
    public int sy;
    public int depth;

    public DoubleBuffer w;
    public DoubleBuffer dw;

    /**
     * > this is how you check if a variable is an array. Oh, Javascript :)
     * Well, I can simply use a dedicated constructor xD
     */
    public Vol(int sx, int sy, int depth, Object c) {
        // we were given dimensions of the vol
        this.sx = sx;
        this.sy = sy;
        this.depth = depth;
        var n = sx * sy * depth;
        this.w = zeros(n);
        this.dw = zeros(n);
        if (c == null) {
            // weight normalization is done to equalize the output
            // variance of every neuron, otherwise neurons with a lot
            // of incoming connections have outputs of larger variance
            var scale = Math.sqrt(1.0 / (sx * sy * depth));
            for (var i = 0; i < n; i++) {
                this.w.set(i, randn(0.0, scale));
            }
        } else if (c instanceof Double cd)
            for (var i = 0; i < n; i++)
                this.w.set(i, cd);
    }

    public Vol(int sx, int sy, int depth) {
        this(sx, sy, depth, null);
    }

    public Vol(double... sx) {
        this(new DoubleBuffer(sx));
    }

    public Vol(DoubleBuffer sx) {

        // we were given a list in sx, assume 1D volume and fill it up
        this.sx = 1;
        this.sy = 1;
        this.depth = sx.size;

        this.w = zeros(this.depth);
        this.dw = zeros(this.depth);
        for (var i = 0; i < this.depth; i++) {
            this.w.set(i, sx.get(i));
        }

    }

    private int calculateIndex(int x, int y, int d) {
        return ((this.sx * y) + x) * this.depth + d;
    }

    public double get(int x, int y, int d) {
        var ix = calculateIndex(x, y, d);
        return this.w.get(ix);
    }

    public void set(int x, int y, int d, double v) {
        var ix = calculateIndex(x, y, d);
        this.w.set(ix, v);
    }

    public void add(int x, int y, int d, double v) {
        var ix = calculateIndex(x, y, d);
        v += this.w.get(ix);
        this.w.set(ix, v);
    }

    public double get_grad(int x, int y, int d) {
        var ix = calculateIndex(x, y, d);
        return this.dw.get(ix);
    }

    public void set_grad(int x, int y, int d, double v) {
        var ix = calculateIndex(x, y, d);
        this.dw.set(ix, v);
    }

    public void add_grad(int x, int y, int d, double v) {
        var ix = calculateIndex(x, y, d);
        v += this.dw.get(ix);
        this.dw.set(ix, v);
    }

    public Vol cloneAndZero() {
        return new Vol(this.sx, this.sy, this.depth, 0.0);
    }

    public Vol clone() {
        var V = new Vol(this.sx, this.sy, this.depth, 0.0);
        var n = this.w.size;
        for (var i = 0; i < n; i++) V.w.set(i, this.w.get(i));
        return V;
    }

    public void addFrom(Vol V) {
        for (var k = 0; k < this.w.size; k++) {
            double v = this.w.get(k);
            v += V.w.get(k);
            this.w.set(k, v);
        }
    }

    public void addFromScaled(Vol V, double a) {
        for (var k = 0; k < this.w.size; k++) {
            double v = this.w.get(k);
            v += V.w.get(k) * a;
            this.w.set(k, v);
        }
    }

    public void setConst(double a) {
        for (var k = 0; k < this.w.size; k++) {
            this.w.set(k, a);
        }
    }

    public String meta() {
        return String.format("Vol {sx = %d, sy = %d, depth = %d}", sx, sy, depth);
    }

    // toJSON: function() {
    // fromJSON: function(json) {

}
















