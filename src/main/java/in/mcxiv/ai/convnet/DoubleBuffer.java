package in.mcxiv.ai.convnet;

import java.util.Arrays;

public class DoubleBuffer {
    public double[] array;
    public int size;
    public int index = 0;

    public DoubleBuffer(DoubleBuffer buffer) {
        this(buffer.array);
    }

    public DoubleBuffer() {
        this(1);
    }

    public DoubleBuffer(int size) {
        this.array = new double[size];
        this.size = size;
    }

    public DoubleBuffer(double...array) {
        this.array = array;
        this.size = array.length;
    }

    public void set(int index, double value) {
        array[index] = value;
    }

    public boolean contains(double value) {
        for (int i = 0; i < array.length; i++)
            if (array[i] == value) return true;
        return false;
    }

    public void add(double value) {
        if (index == size) array = Arrays.copyOf(array, size = array.length + 1);
        array[index++] = value;
    }

    public double get(int i) {
        return array[i];
    }

    public void set(int index, boolean value) {
        set(index, value ? 1 : 0);
    }

    public boolean is(int index, boolean value) {
        return  get(index) == (value ? 1 : 0);
    }

    public void swap(int indexI, int indexJ) {
        double valueI = get(indexI);
        double valueJ = get(indexJ);
        set(indexI, valueJ);
        set(indexJ, valueI);
    }

    public void addValue(int index, double value) {
        array[index] += value;
    }
}

