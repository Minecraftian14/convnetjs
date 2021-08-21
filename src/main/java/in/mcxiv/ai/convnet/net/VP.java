package in.mcxiv.ai.convnet.net;

import java.util.ArrayList;
import java.util.Arrays;

public class VP {

    String[] keys;
    Object[] values;

    public VP(Object... args) {
        assert args.length % 2 == 0;

        keys = new String[args.length / 2];
        values = new Object[args.length / 2];

        for (int i = 0, k = 0; i < args.length; i += 2, k++) {

            keys[k] = args[i].toString();
            values[k] = args[i + 1];

        }
    }

    public boolean is(String key, Object val) {
        for (int i = 0; i < keys.length; i++)
            if (keys[i].equals(key))
                return values[i].equals(val);
        throw throwUp();
    }

    public boolean has(String key) {
        for (int i = 0; i < keys.length; i++)
            if (keys[i].equals(key))
                return true;
        return false;
    }

    public Object getN(String key) {
        for (int i = 0; i < keys.length; i++)
            if (keys[i].equals(key))
                return values[i];
        return null;
    }

    public <T> T getNFC(String key) {
        return (T) getN(key);
    }

    public Object get(String key) {
        for (int i = 0; i < keys.length; i++)
            if (keys[i].equals(key))
                return values[i];
        throw throwUp();
    }

    public <T> T getFC(String key) {
        return (T) get(key);
    }

    public boolean isNull(String key) {
        return getN(key) == null;
    }

    public void add(String key, Object value) {
        keys = Arrays.copyOf(keys, keys.length + 1);
        values = Arrays.copyOf(values, values.length + 1);
        keys[keys.length - 1] = key;
        values[values.length - 1] = value.toString();
    }

    public boolean notNull(String key) {
        return !isNull(key);
    }

    public void set(String key, Object value) {
        for (int i = 0; i < keys.length; i++)
            if (keys[i].equals(key))
                values[i] = value.toString();
    }

    public String getSt(String key) {
        Object obj = get(key);
        if(obj instanceof String s) return s;
        throw throwUp();
    }
    public int getInt(String key) {
        Object obj = get(key);
        if(obj instanceof Double d) return d.intValue();
        if(obj instanceof Integer i) return i;
        if(obj instanceof String s) return (int) Double.parseDouble(s);
        throw throwUp();
    }

    public double getD(String key) {
        Object obj = get(key);
        if(obj instanceof Double d) return d.intValue();
        if(obj instanceof Integer i) return i;
        if(obj instanceof String s) return Double.parseDouble(s);
        throw throwUp();
    }

    private UnsupportedOperationException throwUp() {
        return new UnsupportedOperationException();
    }

    public static class VPL extends ArrayList<VP> {
        public boolean add(Object...args) {
            return super.add(new VP(args));
        }
        public boolean push(Object...args) {
            return add(args);
        }

    }

}
