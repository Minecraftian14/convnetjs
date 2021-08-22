package in.mcxiv.proc;

public class VPUtilities {
    static Class<?> getType(String arg) {
        switch (arg) {
            case "int":
                return Integer.TYPE;
            case "double":
                return Double.TYPE;
            case "String":
                return String.class;
            default:
                throw new AssertionError("No predefined class object defined for type " + arg);
        }
    }

    static String getMethodName(String arg) {
        switch (arg) {
            case "int":
                return "getInt";
            case "double":
                return "getD";
            case "String":
                return "getSt";
            default:
                throw new AssertionError("No predefined class object defined for type " + arg);
        }
    }
}
