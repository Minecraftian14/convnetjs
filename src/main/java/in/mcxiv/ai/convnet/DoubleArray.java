package in.mcxiv.ai.convnet;

import com.badlogic.gdx.math.MathUtils;
import com.badlogic.gdx.utils.StringBuilder;

import java.util.Arrays;

public class DoubleArray {
    public double[] items;
    public int length;
    public boolean ordered;

    /**
     * Creates an ordered array with a capacity of 16.
     */
    public DoubleArray() {
        this(true, 16);
    }

    /**
     * Creates an ordered array with the specified capacity.
     */
    public DoubleArray(int capacity) {
        this(true, capacity);
    }

    /**
     * @param ordered  If false, methods that remove elements may change the order of other elements in the array, which avoids a
     *                 memory copy.
     * @param capacity Any elements added beyond this will cause the backing array to be grown.
     */
    public DoubleArray(boolean ordered, int capacity) {
        this.ordered = ordered;
        items = new double[capacity];
    }

    /**
     * Creates a new array containing the elements in the specific array. The new array will be ordered if the specific array is
     * ordered. The capacity is set to the number of elements, so any subsequent elements added will cause the backing array to be
     * grown.
     */
    public DoubleArray(DoubleArray array) {
        this.ordered = array.ordered;
        length = array.length;
        items = new double[length];
        System.arraycopy(array.items, 0, items, 0, length);
    }

    /**
     * Creates a new ordered array containing the elements in the specified array. The capacity is set to the number of elements,
     * so any subsequent elements added will cause the backing array to be grown.
     */
    public DoubleArray(double[] array) {
        this(true, array, 0, array.length);
    }

    /**
     * Creates a new array containing the elements in the specified array. The capacity is set to the number of elements, so any
     * subsequent elements added will cause the backing array to be grown.
     *
     * @param ordered If false, methods that remove elements may change the order of other elements in the array, which avoids a
     *                memory copy.
     */
    public DoubleArray(boolean ordered, double[] array, int startIndex, int count) {
        this(ordered, count);
        length = count;
        System.arraycopy(array, startIndex, items, 0, count);
    }

    public void add(double value) {
        double[] items = this.items;
        if (length == items.length) items = resize(Math.max(8, (int) (length * 1.75f)));
        items[length++] = value;
    }

    public void add(double value1, double value2) {
        double[] items = this.items;
        if (length + 1 >= items.length) items = resize(Math.max(8, (int) (length * 1.75f)));
        items[length] = value1;
        items[length + 1] = value2;
        length += 2;
    }

    public void add(double value1, double value2, double value3) {
        double[] items = this.items;
        if (length + 2 >= items.length) items = resize(Math.max(8, (int) (length * 1.75f)));
        items[length] = value1;
        items[length + 1] = value2;
        items[length + 2] = value3;
        length += 3;
    }

    public void add(double value1, double value2, double value3, double value4) {
        double[] items = this.items;
        if (length + 3 >= items.length)
            items = resize(Math.max(8, (int) (length * 1.8f))); // 1.75 isn't enough when size=5.
        items[length] = value1;
        items[length + 1] = value2;
        items[length + 2] = value3;
        items[length + 3] = value4;
        length += 4;
    }

    public void addAll(DoubleArray array) {
        addAll(array.items, 0, array.length);
    }

    public void addAll(DoubleArray array, int offset, int length) {
        if (offset + length > array.length)
            throw new IllegalArgumentException("offset + length must be <= size: " + offset + " + " + length + " <= " + array.length);
        addAll(array.items, offset, length);
    }

    public void addAll(double... array) {
        addAll(array, 0, array.length);
    }

    public void addAll(double[] array, int offset, int length) {
        double[] items = this.items;
        int sizeNeeded = this.length + length;
        if (sizeNeeded > items.length) items = resize(Math.max(Math.max(8, sizeNeeded), (int) (this.length * 1.75f)));
        System.arraycopy(array, offset, items, this.length, length);
        this.length += length;
    }

    public double get(int index) {
        if (index >= length) throw new IndexOutOfBoundsException("index can't be >= size: " + index + " >= " + length);
        return items[index];
    }

    public void set(int index, double value) {
        if (index >= length) throw new IndexOutOfBoundsException("index can't be >= size: " + index + " >= " + length);
        items[index] = value;
    }

    public void set(int index, boolean value) {
        set(index, value ? 1 : 0);
    }

    public boolean is(int index, double value) {
        return get(index) == value;
    }

    public boolean is(int index, boolean value) {
        return get(index) == (value ? 1 : 0);
    }

    public void incr(int index, double value) {
        if (index >= length) throw new IndexOutOfBoundsException("index can't be >= size: " + index + " >= " + length);
        items[index] += value;
    }

    public void incr(double value) {
        double[] items = this.items;
        for (int i = 0, n = length; i < n; i++)
            items[i] += value;
    }

    public void mul(int index, double value) {
        if (index >= length) throw new IndexOutOfBoundsException("index can't be >= size: " + index + " >= " + length);
        items[index] *= value;
    }

    public void mul(double value) {
        double[] items = this.items;
        for (int i = 0, n = length; i < n; i++)
            items[i] *= value;
    }

    public void insert(int index, double value) {
        if (index > length) throw new IndexOutOfBoundsException("index can't be > size: " + index + " > " + length);
        double[] items = this.items;
        if (length == items.length) items = resize(Math.max(8, (int) (length * 1.75f)));
        if (ordered)
            System.arraycopy(items, index, items, index + 1, length - index);
        else
            items[length] = items[index];
        length++;
        items[index] = value;
    }

    /**
     * Inserts the specified number of items at the specified index. The new items will have values equal to the values at those
     * indices before the insertion.
     */
    public void insertRange(int index, int count) {
        if (index > length) throw new IndexOutOfBoundsException("index can't be > size: " + index + " > " + length);
        int sizeNeeded = length + count;
        if (sizeNeeded > items.length) items = resize(Math.max(Math.max(8, sizeNeeded), (int) (length * 1.75f)));
        System.arraycopy(items, index, items, index + count, length - index);
        length = sizeNeeded;
    }

    public void swap(int first, int second) {
        if (first >= length) throw new IndexOutOfBoundsException("first can't be >= size: " + first + " >= " + length);
        if (second >= length)
            throw new IndexOutOfBoundsException("second can't be >= size: " + second + " >= " + length);
        double[] items = this.items;
        double firstValue = items[first];
        items[first] = items[second];
        items[second] = firstValue;
    }

    public boolean contains(double value) {
        int i = length - 1;
        double[] items = this.items;
        while (i >= 0)
            if (items[i--] == value) return true;
        return false;
    }

    public int indexOf(double value) {
        double[] items = this.items;
        for (int i = 0, n = length; i < n; i++)
            if (items[i] == value) return i;
        return -1;
    }

    public int lastIndexOf(double value) {
        double[] items = this.items;
        for (int i = length - 1; i >= 0; i--)
            if (items[i] == value) return i;
        return -1;
    }

    public boolean removeValue(double value) {
        double[] items = this.items;
        for (int i = 0, n = length; i < n; i++) {
            if (items[i] == value) {
                removeIndex(i);
                return true;
            }
        }
        return false;
    }

    /**
     * Removes and returns the item at the specified index.
     */
    public double removeIndex(int index) {
        if (index >= length) throw new IndexOutOfBoundsException("index can't be >= size: " + index + " >= " + length);
        double[] items = this.items;
        double value = items[index];
        length--;
        if (ordered)
            System.arraycopy(items, index + 1, items, index, length - index);
        else
            items[index] = items[length];
        return value;
    }

    /**
     * Removes the items between the specified indices, inclusive.
     */
    public void removeRange(int start, int end) {
        int n = length;
        if (end >= n) throw new IndexOutOfBoundsException("end can't be >= size: " + end + " >= " + length);
        if (start > end) throw new IndexOutOfBoundsException("start can't be > end: " + start + " > " + end);
        int count = end - start + 1, lastIndex = n - count;
        if (ordered)
            System.arraycopy(items, start + count, items, start, n - (start + count));
        else {
            int i = Math.max(lastIndex, end + 1);
            System.arraycopy(items, i, items, start, n - i);
        }
        length = n - count;
    }

    /**
     * Removes from this array all of elements contained in the specified array.
     *
     * @return true if this array was modified.
     */
    public boolean removeAll(DoubleArray array) {
        int size = this.length;
        int startSize = size;
        double[] items = this.items;
        for (int i = 0, n = array.length; i < n; i++) {
            double item = array.get(i);
            for (int ii = 0; ii < size; ii++) {
                if (item == items[ii]) {
                    removeIndex(ii);
                    size--;
                    break;
                }
            }
        }
        return size != startSize;
    }

    /**
     * Removes and returns the last item.
     */
    public double pop() {
        return items[--length];
    }

    /**
     * Returns the last item.
     */
    public double peek() {
        return items[length - 1];
    }

    /**
     * Returns the first item.
     */
    public double first() {
        if (length == 0) throw new IllegalStateException("Array is empty.");
        return items[0];
    }

    /**
     * Returns true if the array has one or more items.
     */
    public boolean notEmpty() {
        return length > 0;
    }

    /**
     * Returns true if the array is empty.
     */
    public boolean isEmpty() {
        return length == 0;
    }

    public void clear() {
        length = 0;
    }

    /**
     * Reduces the size of the backing array to the size of the actual items. This is useful to release memory when many items
     * have been removed, or if it is known that more items will not be added.
     *
     * @return {@link #items}
     */
    public double[] shrink() {
        if (items.length != length) resize(length);
        return items;
    }

    /**
     * Increases the size of the backing array to accommodate the specified number of additional items. Useful before adding many
     * items to avoid multiple backing array resizes.
     *
     * @return {@link #items}
     */
    public double[] ensureCapacity(int additionalCapacity) {
        if (additionalCapacity < 0)
            throw new IllegalArgumentException("additionalCapacity must be >= 0: " + additionalCapacity);
        int sizeNeeded = length + additionalCapacity;
        if (sizeNeeded > items.length) resize(Math.max(Math.max(8, sizeNeeded), (int) (length * 1.75f)));
        return items;
    }

    /**
     * Sets the array size, leaving any values beyond the current size undefined.
     *
     * @return {@link #items}
     */
    public double[] setSize(int newSize) {
        if (newSize < 0) throw new IllegalArgumentException("newSize must be >= 0: " + newSize);
        if (newSize > items.length) resize(Math.max(8, newSize));
        length = newSize;
        return items;
    }

    protected double[] resize(int newSize) {
        double[] newItems = new double[newSize];
        double[] items = this.items;
        System.arraycopy(items, 0, newItems, 0, Math.min(length, newItems.length));
        this.items = newItems;
        return newItems;
    }

    public void sort() {
        Arrays.sort(items, 0, length);
    }

    public void reverse() {
        double[] items = this.items;
        for (int i = 0, lastIndex = length - 1, n = length / 2; i < n; i++) {
            int ii = lastIndex - i;
            double temp = items[i];
            items[i] = items[ii];
            items[ii] = temp;
        }
    }

    public void shuffle() {
        double[] items = this.items;
        for (int i = length - 1; i >= 0; i--) {
            int ii = MathUtils.random(i);
            double temp = items[i];
            items[i] = items[ii];
            items[ii] = temp;
        }
    }

    /**
     * Reduces the size of the array to the specified size. If the array is already smaller than the specified size, no action is
     * taken.
     */
    public void truncate(int newSize) {
        if (length > newSize) length = newSize;
    }

    /**
     * Returns a random item from the array, or zero if the array is empty.
     */
    public double random() {
        if (length == 0) return 0;
        return items[MathUtils.random(0, length - 1)];
    }

    public double[] toArray() {
        double[] array = new double[length];
        System.arraycopy(items, 0, array, 0, length);
        return array;
    }

    public int hashCode() {
        if (!ordered) return super.hashCode();
        double[] items = this.items;
        long h = 1;
        for (int i = 0, n = length; i < n; i++)
//            h = h * 31 + Double.doubleToLongBits(items[i]);
            // TODO is that right?
            h = h * 63 + Double.doubleToLongBits(items[i]);
        return Long.hashCode(h);
    }


    /**
     * Returns false if either array is unordered.
     */
    public boolean equals(Object object) {
        if (object == this) return true;
        if (!ordered) return false;
        if (!(object instanceof DoubleArray)) return false;
        DoubleArray array = (DoubleArray) object;
        if (!array.ordered) return false;
        int n = length;
        if (n != array.length) return false;
        double[] items1 = this.items, items2 = array.items;
        for (int i = 0; i < n; i++)
            if (items1[i] != items2[i]) return false;
        return true;
    }

    /**
     * Returns false if either array is unordered.
     */
    public boolean equals(Object object, double epsilon) {
        if (object == this) return true;
        if (!(object instanceof DoubleArray)) return false;
        DoubleArray array = (DoubleArray) object;
        int n = length;
        if (n != array.length) return false;
        if (!ordered) return false;
        if (!array.ordered) return false;
        double[] items1 = this.items, items2 = array.items;
        for (int i = 0; i < n; i++)
            if (Math.abs(items1[i] - items2[i]) > epsilon) return false;
        return true;
    }

    public String toString() {
        if (length == 0) return "[]";
        double[] items = this.items;
        StringBuilder buffer = new StringBuilder(32);
        buffer.append('[');
        buffer.append(items[0]);
        for (int i = 1; i < length; i++) {
            buffer.append(", ");
            buffer.append(items[i]);
        }
        buffer.append(']');
        return buffer.toString();
    }

    public String toString(String separator) {
        if (length == 0) return "";
        double[] items = this.items;
        StringBuilder buffer = new StringBuilder(32);
        buffer.append(items[0]);
        for (int i = 1; i < length; i++) {
            buffer.append(separator);
            buffer.append(items[i]);
        }
        return buffer.toString();
    }

    /**
     * @see #DoubleArray(double[])
     */
    static public DoubleArray with(double... array) {
        return new DoubleArray(array);
    }

    public void addValue(int i, double value) {
        set(i, get(i) + value);
    }
}

