package in.mcxiv.proc;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.stream.Stream;

public class BasicUtilities {

    static String[] removeEveryThird(String[] args) {
        String[] new_args = new String[(args.length / 3) * 2];
        if (new_args.length == 0) return new_args;
        for (int i = 0, k = 0; i < args.length; i += 3, k += 2) {
            new_args[k] = args[i];
            new_args[k + 1] = args[i + 1];
        }
        return new_args;
    }

    static String[] onlyThird(String[] args) {
        String[] new_args = new String[(args.length / 3)];
        for (int i = 2, k = 0; i < args.length; i += 3, k++) {
            new_args[k] = args[i];
        }
        return new_args;
    }

    static <T> T[] concat(T[] a1, T[] a2) {
        return Stream.concat(Arrays.stream(a1), Arrays.stream(a2))
                .toArray(size -> (T[]) Array.newInstance(a1.getClass().getComponentType(), size));
    }

    @SafeVarargs
    static <T> T[] concat(T[]... a) {
        T[] t = concat(a[0], a[1]);
        for (int i = 2; i < a.length; i++) {
            t = concat(t, a[i]);
        }
        return t;
    }

    static String[] splitIntoAppropriateArray(String args) {
        return args
                .replace("'", "\"")
                .replace(",", " ")
                .replaceAll("[ ][ ]+", " ")
                .split(" ");
    }
}
